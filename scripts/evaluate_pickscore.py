#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import wilcoxon
from transformers import AutoModel, AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate paired image directories with the held-out PickScore model."
    )
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--method-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="yuvalkirstain/PickScore_v1")
    parser.add_argument("--processor", default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260814)
    return parser.parse_args()


def score_pairs(model, processor, baseline_images, method_images, prompts, device):
    image_inputs = processor(images=baseline_images + method_images, return_tensors="pt").to(device)
    text_inputs = processor(
        text=prompts, padding=True, truncation=True, max_length=77, return_tensors="pt"
    ).to(device)
    with torch.inference_mode():
        image_features = model.get_image_features(**image_inputs)
        text_features = model.get_text_features(**text_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        batch_size = len(prompts)
        scale = model.logit_scale.exp()
        baseline_scores = scale * (text_features * image_features[:batch_size]).sum(dim=1)
        method_scores = scale * (text_features * image_features[batch_size:]).sum(dim=1)
        return baseline_scores.cpu().tolist(), method_scores.cpu().tolist()


def main():
    args = parse_args()
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(args.processor)
    model = AutoModel.from_pretrained(args.model).eval().to(device)
    rows = []
    items = list(mapping.items())
    for start in range(0, len(items), args.batch_size):
        batch = items[start : start + args.batch_size]
        baseline_images = []
        method_images = []
        prompts = []
        for _, record in batch:
            relative = Path(record["image_path"])
            with Image.open(args.baseline_root / relative) as image:
                baseline_images.append(image.convert("RGB"))
            with Image.open(args.method_root / relative) as image:
                method_images.append(image.convert("RGB"))
            prompts.append(record["editing_prompt"].replace("[", "").replace("]", ""))
        baseline_scores, method_scores = score_pairs(
            model, processor, baseline_images, method_images, prompts, device
        )
        for (file_id, record), baseline_score, method_score in zip(
            batch, baseline_scores, method_scores
        ):
            rows.append(
                {
                    "file_id": file_id,
                    "category": Path(record["image_path"]).parts[0],
                    "baseline_pickscore": baseline_score,
                    "method_pickscore": method_score,
                    "improvement": method_score - baseline_score,
                }
            )
        print(f"scored {len(rows)}/{len(mapping)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    improvements = np.array([row["improvement"] for row in rows])
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(
        0, len(improvements), size=(args.bootstrap_resamples, len(improvements))
    )
    ci = np.quantile(improvements[indices].mean(axis=1), [0.025, 0.975])
    nonzero = improvements[np.abs(improvements) > 1e-12]
    summary = {
        "paired_samples": len(rows),
        "mean_baseline": float(np.mean([row["baseline_pickscore"] for row in rows])),
        "mean_method": float(np.mean([row["method_pickscore"] for row in rows])),
        "mean_improvement": float(improvements.mean()),
        "bootstrap_ci95": [float(ci[0]), float(ci[1])],
        "wins": int((improvements > 1e-12).sum()),
        "ties": int((np.abs(improvements) <= 1e-12).sum()),
        "losses": int((improvements < -1e-12).sum()),
        "wilcoxon_greater_p": float(wilcoxon(nonzero, alternative="greater").pvalue),
        "model": args.model,
        "processor": args.processor,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
