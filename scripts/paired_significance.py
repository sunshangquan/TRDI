#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon


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
    parser = argparse.ArgumentParser(
        description="Run paired significance tests on two PIE-Bench evaluation CSVs."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["file_id"]: row
            for row in csv.DictReader(handle)
            if row["file_id"] != "Avg"
        }


def holm_adjust(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running_maximum = 0.0
    for rank, index in enumerate(order):
        corrected = min(1.0, (len(pvalues) - rank) * pvalues[index])
        running_maximum = max(running_maximum, corrected)
        adjusted[index] = running_maximum
    return adjusted


def bootstrap_mean_interval(values, resamples, rng):
    means = np.empty(resamples, dtype=np.float64)
    batch_size = 1_000
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return np.quantile(means, [0.025, 0.975])


def analyze(baseline_path, method_path, bootstrap_resamples, seed):
    baseline = read_rows(baseline_path)
    method = read_rows(method_path)
    if set(baseline) != set(method):
        missing_method = sorted(set(baseline) - set(method))
        missing_baseline = sorted(set(method) - set(baseline))
        raise ValueError(
            "paired CSV file IDs differ: "
            f"missing from method={missing_method[:5]}, "
            f"missing from baseline={missing_baseline[:5]}"
        )
    file_ids = sorted(baseline)
    rng = np.random.default_rng(seed)
    results = []
    raw_pvalues = []
    for metric, higher_is_better in METRICS:
        baseline_values = np.array(
            [float(baseline[file_id][metric]) for file_id in file_ids], dtype=np.float64
        )
        method_values = np.array(
            [float(method[file_id][metric]) for file_id in file_ids], dtype=np.float64
        )
        finite = np.isfinite(baseline_values) & np.isfinite(method_values)
        baseline_values = baseline_values[finite]
        method_values = method_values[finite]
        improvement = (
            method_values - baseline_values
            if higher_is_better
            else baseline_values - method_values
        )
        ci_low, ci_high = bootstrap_mean_interval(
            improvement, bootstrap_resamples, rng
        )
        tolerance = 1e-12
        wins = int((improvement > tolerance).sum())
        losses = int((improvement < -tolerance).sum())
        ties = int(len(improvement) - wins - losses)
        nonzero = improvement[np.abs(improvement) > tolerance]
        wilcoxon_p = (
            float(wilcoxon(nonzero, alternative="greater").pvalue)
            if len(nonzero)
            else 1.0
        )
        sign_p = (
            float(binomtest(wins, wins + losses, 0.5, alternative="greater").pvalue)
            if wins + losses
            else 1.0
        )
        results.append(
            {
                "metric": metric,
                "higher_is_better": higher_is_better,
                "paired_samples": int(len(improvement)),
                "mean_improvement": float(improvement.mean()),
                "bootstrap_ci95": [float(ci_low), float(ci_high)],
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "wilcoxon_p": wilcoxon_p,
                "sign_test_p": sign_p,
            }
        )
        raw_pvalues.append(wilcoxon_p)

    for result, adjusted_p in zip(results, holm_adjust(raw_pvalues)):
        result["wilcoxon_holm_p"] = float(adjusted_p)
        result["bootstrap_mean_positive"] = bool(result["bootstrap_ci95"][0] > 0)
        result["wilcoxon_holm_significant"] = bool(adjusted_p < 0.05)

    return {
        "baseline": str(baseline_path.resolve()),
        "method": str(method_path.resolve()),
        "seed": seed,
        "bootstrap_resamples": bootstrap_resamples,
        "paired_file_ids": len(file_ids),
        "bootstrap_positive_metrics": int(
            sum(result["bootstrap_mean_positive"] for result in results)
        ),
        "wilcoxon_holm_significant_metrics": int(
            sum(result["wilcoxon_holm_significant"] for result in results)
        ),
        "metrics": results,
    }


def main():
    args = parse_args()
    result = analyze(
        args.baseline,
        args.method,
        args.bootstrap_resamples,
        args.seed,
    )
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
