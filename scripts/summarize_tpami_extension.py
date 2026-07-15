#!/usr/bin/env python3

import argparse
import collections
import csv
import json
import math
from pathlib import Path


METRICS = [
    ("structure_distance", False),
    ("psnr_unedit_part", True),
    ("lpips_unedit_part", False),
    ("mse_unedit_part", False),
    ("ssim_unedit_part", True),
    ("clip_similarity_target_image", True),
    ("clip_similarity_target_image_edit_part", True),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping_file", required=True)
    parser.add_argument("--result_path", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_markdown")
    return parser.parse_args()


def float_or_nan(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def nanmean(values):
    clean = [value for value in values if math.isfinite(value)]
    return sum(clean) / len(clean) if clean else math.nan


def load_rows(result_path, method):
    csv_path = Path(result_path) / f"{method}.csv"
    rows = []
    with csv_path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("file_id") != "Avg":
                rows.append(row)
    return rows


def summarize_rows(rows):
    return {
        metric: nanmean(float_or_nan(row.get(metric)) for row in rows)
        for metric, _ in METRICS
    }


def metric_wins(method_summary, baseline_summary):
    wins = 0
    deltas = {}
    for metric, higher_is_better in METRICS:
        delta = method_summary[metric] - baseline_summary[metric]
        good = delta > 0 if higher_is_better else delta < 0
        wins += int(good)
        deltas[metric] = {
            "baseline": baseline_summary[metric],
            "method": method_summary[metric],
            "delta": delta,
            "good": good,
        }
    return wins, deltas


def category_summaries(rows_by_method, mapping, baseline):
    categories = collections.defaultdict(list)
    for row in rows_by_method[baseline]:
        file_id = row["file_id"]
        if file_id in mapping:
            categories[str(mapping[file_id]["editing_type_id"])].append(file_id)

    rows_by_id = {
        method: {row["file_id"]: row for row in rows}
        for method, rows in rows_by_method.items()
    }
    out = {}
    for method in rows_by_method:
        if method == baseline:
            continue
        total = 0
        per_category = {}
        for category, file_ids in sorted(categories.items(), key=lambda item: int(item[0])):
            baseline_rows = [rows_by_id[baseline][file_id] for file_id in file_ids]
            method_rows = [rows_by_id[method][file_id] for file_id in file_ids]
            baseline_summary = summarize_rows(baseline_rows)
            method_summary = summarize_rows(method_rows)
            wins, deltas = metric_wins(method_summary, baseline_summary)
            total += wins
            per_category[category] = {
                "samples": len(file_ids),
                "wins": wins,
                "metrics": deltas,
            }
        out[method] = {
            "category_metric_wins": total,
            "category_metric_total": len(categories) * len(METRICS),
            "categories": per_category,
        }
    return out


def selection_counts(image_root, methods):
    out = {}
    for method in methods:
        path = Path(image_root) / method / "selection_records.json"
        if not path.is_file():
            continue
        records = json.loads(path.read_text(encoding="utf-8"))
        counter = collections.Counter(record["selected_schedule_mode"] for record in records)
        out[method] = {
            "total": len(records),
            "counts": dict(sorted(counter.items())),
        }
    return out


def markdown_table(summary, baseline, methods):
    headers = [
        "Method",
        "Structure Distance ↓",
        "PSNR ↑",
        "LPIPS ↓",
        "MSE ↓",
        "SSIM ↑",
        "CLIP Target ↑",
        "CLIP Edit-Part ↑",
        "Wins vs Baseline",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    baseline_summary = summary["aggregate"][baseline]
    for method in methods:
        values = summary["aggregate"][method]
        wins = "-" if method == baseline else str(summary["comparison_to_baseline"][method]["wins"]) + "/7"
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    f"{values['structure_distance']:.6f}",
                    f"{values['psnr_unedit_part']:.4f}",
                    f"{values['lpips_unedit_part']:.6f}",
                    f"{values['mse_unedit_part']:.6f}",
                    f"{values['ssim_unedit_part']:.6f}",
                    f"{values['clip_similarity_target_image']:.4f}",
                    f"{values['clip_similarity_target_image_edit_part']:.4f}",
                    wins,
                ]
            )
            + " |"
        )
    if baseline_summary:
        return "\n".join(lines) + "\n"
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    mapping = json.loads(Path(args.mapping_file).read_text(encoding="utf-8"))
    methods = args.methods
    if args.baseline not in methods:
        methods = [args.baseline, *methods]

    rows_by_method = {method: load_rows(args.result_path, method) for method in methods}
    aggregate = {method: summarize_rows(rows) for method, rows in rows_by_method.items()}
    comparison = {}
    for method in methods:
        if method == args.baseline:
            continue
        wins, deltas = metric_wins(aggregate[method], aggregate[args.baseline])
        comparison[method] = {"wins": wins, "metrics": deltas}

    summary = {
        "baseline": args.baseline,
        "methods": methods,
        "aggregate": aggregate,
        "comparison_to_baseline": comparison,
        "category": category_summaries(rows_by_method, mapping, args.baseline),
        "selection_counts": selection_counts(args.image_root, methods),
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.output_markdown:
        Path(args.output_markdown).write_text(
            markdown_table(summary, args.baseline, methods),
            encoding="utf-8",
        )
    print(f"summary={output_json}")


if __name__ == "__main__":
    main()
