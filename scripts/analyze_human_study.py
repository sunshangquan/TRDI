#!/usr/bin/env python3

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


OUTCOMES = ("edit_choice", "preserve_choice", "overall_choice")


def parse_args():
    parser = argparse.ArgumentParser(description="Unblind and analyze preference responses.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--responses", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260814)
    return parser.parse_args()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def method_outcome(choice, manifest):
    if choice == "Tie":
        return 0.5
    if choice not in ("Left", "Right"):
        raise ValueError(f"invalid choice: {choice!r}")
    return 1.0 if manifest[f"{choice.lower()}_method"] == "CGA-TRDI" else 0.0


def bootstrap_by_case(case_scores, resamples, rng):
    case_ids = sorted(case_scores)
    values = np.array([np.mean(case_scores[case_id]) for case_id in case_ids])
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def main():
    args = parse_args()
    manifest_rows = read_csv(args.manifest)
    manifest = {(row["participant_id"], row["trial_id"]): row for row in manifest_rows}
    responses = [row for path in args.responses for row in read_csv(path)]
    rng = np.random.default_rng(args.seed)
    attention_rows = [row for row in responses if row.get("is_attention_check", "0") == "1"]
    attention_passed = sum(
        all(row[field] == "Tie" for field in OUTCOMES) for row in attention_rows
    )
    result = {
        "responses": len(responses),
        "unique_participants": 0,
        "quality_control": {
            "attention_checks": len(attention_rows),
            "attention_checks_passed": attention_passed,
            "attention_pass_rate": attention_passed / len(attention_rows)
            if attention_rows
            else None,
        },
        "outcomes": {},
    }
    result["unique_participants"] = len({row["participant_id"] for row in responses})
    for field in OUTCOMES:
        case_scores = defaultdict(list)
        wins = ties = losses = 0
        for row in responses:
            key = (row["participant_id"], row["trial_id"])
            if key not in manifest:
                raise KeyError(f"response has no manifest row: {key}")
            if row.get("is_repeat", "0") == "1":
                continue
            if row.get("is_attention_check", "0") == "1":
                continue
            score = method_outcome(row[field], manifest[key])
            case_scores[row["case_id"]].append(score)
            wins += score == 1.0
            ties += score == 0.5
            losses += score == 0.0
        total = wins + ties + losses
        result["outcomes"][field] = {
            "cga_wins": wins,
            "ties": ties,
            "cga_losses": losses,
            "cga_preference_with_half_ties": (wins + 0.5 * ties) / total,
            "bootstrap_ci95_by_case": bootstrap_by_case(
                case_scores, args.bootstrap_resamples, rng
            ),
            "unique_cases": len(case_scores),
        }
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
