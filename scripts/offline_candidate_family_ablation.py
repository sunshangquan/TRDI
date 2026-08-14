#!/usr/bin/env python3

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from offline_selector_sensitivity import METRICS, choose, mean_metrics, parse_named_paths, read_metric_csv


def parse_families(value):
    result = []
    for item in value.split(","):
        name, separator, schedules = item.partition("=")
        if not separator:
            raise argparse.ArgumentTypeError(f"expected name=schedule/schedule, got {item!r}")
        result.append((name, schedules.split("/")))
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ablate saved candidate families without regenerating images."
    )
    parser.add_argument("--selection-records", type=Path, required=True)
    parser.add_argument("--candidate-csvs", type=parse_named_paths, required=True)
    parser.add_argument(
        "--families",
        type=parse_families,
        default=parse_families(
            "TRDI=trdi,Late=trdi/adaptive_late,"
            "F50+Late=trdi/adaptive_noise_floor50/adaptive_late,"
            "All=trdi/adaptive_noise_floor50/adaptive_noise_floor75/adaptive_late"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    records = json.loads(args.selection_records.read_text(encoding="utf-8"))
    metric_rows = {
        schedule: read_metric_csv(path) for schedule, path in args.candidate_csvs.items()
    }
    baseline_selected = [
        (Path(record["file"]).stem, "trdi", Path(record["file"]).parts[0])
        for record in records
    ]
    baseline = mean_metrics(baseline_selected, metric_rows)
    results = []
    for name, schedules in args.families:
        selected = []
        counts = Counter()
        for record in records:
            candidates = [
                candidate
                for candidate in record["candidates"]
                if candidate["schedule_mode"] in schedules
            ]
            missing = set(schedules) - {candidate["schedule_mode"] for candidate in candidates}
            if missing:
                raise ValueError(f"record {record['file']} lacks schedules {sorted(missing)}")
            selected_index = choose(candidates, (1.0, 0.4, 0.2), 5, 3)
            schedule = candidates[selected_index]["schedule_mode"]
            category = Path(record["file"]).parts[0]
            selected.append((Path(record["file"]).stem, schedule, category))
            counts[schedule] += 1
        aggregate = mean_metrics(selected, metric_rows)
        aggregate_wins = sum(
            aggregate[metric] > baseline[metric] if higher else aggregate[metric] < baseline[metric]
            for metric, higher in METRICS
        )
        by_category = defaultdict(list)
        baseline_by_category = defaultdict(list)
        for item in selected:
            by_category[item[2]].append(item)
        for item in baseline_selected:
            baseline_by_category[item[2]].append(item)
        category_wins = 0
        for category in by_category:
            category_result = mean_metrics(by_category[category], metric_rows)
            category_baseline = mean_metrics(baseline_by_category[category], metric_rows)
            category_wins += sum(
                category_result[metric] > category_baseline[metric]
                if higher
                else category_result[metric] < category_baseline[metric]
                for metric, higher in METRICS
            )
        results.append(
            {
                "family": name,
                "schedules": schedules,
                "switched": len(records) - counts["trdi"],
                "aggregate_wins": aggregate_wins,
                "category_wins": category_wins,
                "schedule_counts": dict(counts),
                "metrics": aggregate,
            }
        )
    output = {"baseline": baseline, "families": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
