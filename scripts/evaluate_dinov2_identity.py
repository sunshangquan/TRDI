#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import wilcoxon
from transformers import AutoImageProcessor, AutoModel


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate source-identity preservation with a held-out DINOv2 model."
        )
    )
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--method-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="facebook/dinov2-small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260814)
    return parser.parse_args()


def encode_images(model, processor, images, device):
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.inference_mode():
        features = model(**inputs).last_hidden_state[:, 0]
        features = features / features.norm(dim=-1, keepdim=True)
    return features


def score_triplets(model, processor, source_images, baseline_images, method_images, device):
    batch_size = len(source_images)
    features = encode_images(
        model,
        processor,
        source_images + baseline_images + method_images,
        device,
    )
    source = features[:batch_size]
    baseline = features[batch_size : 2 * batch_size]
    method = features[2 * batch_size :]
    baseline_scores = (source * baseline).sum(dim=1)
    method_scores = (source * method).sum(dim=1)
    return baseline_scores.cpu().tolist(), method_scores.cpu().tolist()


def summarize_improvements(improvements, bootstrap_resamples, seed):
    improvements = np.asarray(improvements, dtype=np.float64)
    if improvements.ndim != 1 or len(improvements) == 0:
        raise ValueError("improvements must be a non-empty one-dimensional array")
    rng = np.random.default_rng(seed)
    means = np.empty(bootstrap_resamples, dtype=np.float64)
    batch_size = 1_000
    for start in range(0, bootstrap_resamples, batch_size):
        stop = min(start + batch_size, bootstrap_resamples)
        indices = rng.integers(
            0, len(improvements), size=(stop - start, len(improvements))
        )
        means[start:stop] = improvements[indices].mean(axis=1)
    ci = np.quantile(means, [0.025, 0.975])
    tolerance = 1e-12
    nonzero = improvements[np.abs(improvements) > tolerance]
    return {
        "mean_improvement": float(improvements.mean()),
        "bootstrap_ci95": [float(ci[0]), float(ci[1])],
        "wins": int((improvements > tolerance).sum()),
        "ties": int((np.abs(improvements) <= tolerance).sum()),
        "losses": int((improvements < -tolerance).sum()),
        "wilcoxon_greater_p": (
            float(wilcoxon(nonzero, alternative="greater").pvalue)
            if len(nonzero)
            else 1.0
        ),
    }


def main():
    args = parse_args()
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).eval().to(device)
    rows = []
    items = list(mapping.items())
    for start in range(0, len(items), args.batch_size):
        batch = items[start : start + args.batch_size]
        source_images = []
        baseline_images = []
        method_images = []
        for _, record in batch:
            relative = Path(record["image_path"])
            with Image.open(args.source_root / relative) as image:
                source_images.append(image.convert("RGB"))
            with Image.open(args.baseline_root / relative) as image:
                baseline_images.append(image.convert("RGB"))
            with Image.open(args.method_root / relative) as image:
                method_images.append(image.convert("RGB"))
        baseline_scores, method_scores = score_triplets(
            model,
            processor,
            source_images,
            baseline_images,
            method_images,
            device,
        )
        for (file_id, record), baseline_score, method_score in zip(
            batch, baseline_scores, method_scores
        ):
            rows.append(
                {
                    "file_id": file_id,
                    "category": Path(record["image_path"]).parts[0],
                    "baseline_dinov2_identity": baseline_score,
                    "method_dinov2_identity": method_score,
                    "improvement": method_score - baseline_score,
                }
            )
        print(f"scored {len(rows)}/{len(mapping)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "paired_samples": len(rows),
        "mean_baseline": float(
            np.mean([row["baseline_dinov2_identity"] for row in rows])
        ),
        "mean_method": float(
            np.mean([row["method_dinov2_identity"] for row in rows])
        ),
        **summarize_improvements(
            [row["improvement"] for row in rows],
            args.bootstrap_resamples,
            args.seed,
        ),
        "model": args.model,
        "uses_selector_signal": False,
        "uses_edit_mask": False,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
