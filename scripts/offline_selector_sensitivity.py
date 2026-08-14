#!/usr/bin/env python3

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


METRICS = (
    ("structure_distance", False),
    ("psnr_unedit_part", True),
    ("lpips_unedit_part", False),
    ("mse_unedit_part", False),
    ("ssim_unedit_part", True),
    ("clip_similarity_target_image", True),
    ("clip_similarity_target_image_edit_part", True),
)


def parse_named_paths(value):
    result = {}
    for item in value.split(","):
        name, separator, path = item.partition("=")
        if not separator:
            raise argparse.ArgumentTypeError(f"expected name=path, got {item!r}")
        result[name] = Path(path)
    return result


def parse_weights(value):
    configurations = []
    for item in value.split(","):
        name, separator, numbers = item.partition("=")
        if not separator:
            raise argparse.ArgumentTypeError(f"expected name=clip/preserve/structure")
        weights = tuple(float(number) for number in numbers.split("/"))
        if len(weights) != 3:
            raise argparse.ArgumentTypeError(f"expected three weights in {item!r}")
        configurations.append((name, weights))
    return configurations


def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-rank saved candidate features and aggregate existing metric CSVs."
    )
    parser.add_argument("--selection-records", type=Path, required=True)
    parser.add_argument("--candidate-csvs", type=parse_named_paths, required=True)
    parser.add_argument(
        "--weights",
        type=parse_weights,
        default=parse_weights("light=1/.2/.1,default=1/.4/.2,heavy=1/.6/.3"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-preserve-wins", type=int, default=5)
    parser.add_argument("--min-preserve-wins-for-edit", type=int, default=3)
    return parser.parse_args()


def normalize(values, higher=True):
    values = np.asarray(values, dtype=np.float64)
    span = values.max() - values.min()
    normalized = np.full_like(values, 0.5) if span <= 1e-12 else (values - values.min()) / span
    return normalized if higher else 1.0 - normalized


def choose(candidates, weights, min_preserve_wins, min_preserve_wins_for_edit):
    feature = lambda key: [candidate["features"][key] for candidate in candidates]
    clip = normalize(feature("target_clip_edit"))
    preserve = np.mean(
        [
            normalize(feature("preserve_psnr")),
            normalize(feature("preserve_ssim")),
            normalize(feature("preserve_lpips"), False),
            normalize(feature("preserve_mse"), False),
        ],
        axis=0,
    )
    structure = normalize(feature("structure_distance"), False)
    scores = weights[0] * clip + weights[1] * preserve + weights[2] * structure
    baseline_index = next(
        index for index, candidate in enumerate(candidates) if candidate["schedule_mode"] == "trdi"
    )
    baseline = candidates[baseline_index]["features"]
    eligible = np.zeros(len(candidates), dtype=bool)
    eligible[baseline_index] = True
    for index, candidate in enumerate(candidates):
        if index == baseline_index:
            continue
        features = candidate["features"]
        preserve_wins = sum(
            (
                features["preserve_psnr"] > baseline["preserve_psnr"],
                features["preserve_ssim"] > baseline["preserve_ssim"],
                features["preserve_lpips"] < baseline["preserve_lpips"],
                features["preserve_mse"] < baseline["preserve_mse"],
                features["structure_distance"] < baseline["structure_distance"],
            )
        )
        semantic_ok = (
            features["target_clip"] >= baseline["target_clip"]
            and features["target_clip_edit"] >= baseline["target_clip_edit"]
        )
        improves_edit = (
            features["target_clip_edit"] > baseline["target_clip_edit"]
            and preserve_wins >= min_preserve_wins_for_edit
        )
        preserves = preserve_wins >= min_preserve_wins
        eligible[index] = semantic_ok and (improves_edit or preserves)
    return int(np.argmax(np.where(eligible, scores, -np.inf)))


def read_metric_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["file_id"]: row
            for row in csv.DictReader(handle)
            if row["file_id"] != "Avg"
        }


def mean_metrics(selected, metric_rows):
    return {
        metric: float(
            np.nanmean(
                [float(metric_rows[schedule][file_id][metric]) for file_id, schedule, _ in selected]
            )
        )
        for metric, _ in METRICS
    }


def main():
    args = parse_args()
    records = json.loads(args.selection_records.read_text(encoding="utf-8"))
    metric_rows = {
        schedule: read_metric_csv(path) for schedule, path in args.candidate_csvs.items()
    }
    baseline_metrics = mean_metrics(
        [(Path(record["file"]).stem, "trdi", Path(record["file"]).parts[0]) for record in records],
        metric_rows,
    )
    results = []
    for name, weights in args.weights:
        selected = []
        counts = Counter()
        for record in records:
            index = choose(
                record["candidates"],
                weights,
                args.min_preserve_wins,
                args.min_preserve_wins_for_edit,
            )
            schedule = record["candidates"][index]["schedule_mode"]
            file_id = Path(record["file"]).stem
            category = Path(record["file"]).parts[0]
            selected.append((file_id, schedule, category))
            counts[schedule] += 1
        aggregate = mean_metrics(selected, metric_rows)
        aggregate_wins = sum(
            aggregate[metric] > baseline_metrics[metric]
            if higher
            else aggregate[metric] < baseline_metrics[metric]
            for metric, higher in METRICS
        )
        category_wins = 0
        by_category = defaultdict(list)
        for item in selected:
            by_category[item[2]].append(item)
        baseline_by_category = defaultdict(list)
        for record in records:
            category = Path(record["file"]).parts[0]
            baseline_by_category[category].append((Path(record["file"]).stem, "trdi", category))
        for category, category_selected in by_category.items():
            selected_metrics = mean_metrics(category_selected, metric_rows)
            category_baseline = mean_metrics(baseline_by_category[category], metric_rows)
            category_wins += sum(
                selected_metrics[metric] > category_baseline[metric]
                if higher
                else selected_metrics[metric] < category_baseline[metric]
                for metric, higher in METRICS
            )
        results.append(
            {
                "name": name,
                "weights": weights,
                "switched": len(records) - counts["trdi"],
                "schedule_counts": dict(counts),
                "aggregate_wins": aggregate_wins,
                "category_wins": category_wins,
                "metrics": aggregate,
            }
        )
    output = {"baseline": baseline_metrics, "configurations": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
