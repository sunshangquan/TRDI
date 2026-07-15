#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import subprocess
import time
from pathlib import Path


CASE_TRDI = "sdxl_ddim_trdi_full700_tpami_v1"
CASE_FLOOR50 = "sdxl_ddim_adaptive_noise_floor50_full700_tpami_v1"
CASE_FLOOR75 = "sdxl_ddim_adaptive_noise_floor75_full700_tpami_v1"
CASE_LATE = "sdxl_ddim_adaptive_late_full700_tpami_v1"

ABLATIONS = {
    "sdxl_ddim_select_score4_full700_tpami_v1": {
        "selection_policy": "score",
        "selection_min_preserve_wins_for_edit": 0,
        "selection_edit_clip_fraction": 1.0,
        "selection_target_clip_drop_tolerance": 0.0,
        "selection_edit_clip_drop_tolerance": 0.0,
    },
    "sdxl_ddim_select_balanced4_full700_tpami_v1": {
        "selection_policy": "dominance",
        "selection_min_preserve_wins_for_edit": 2,
        "selection_edit_clip_fraction": 1.0,
        "selection_target_clip_drop_tolerance": 0.0,
        "selection_edit_clip_drop_tolerance": 0.0,
    },
    "sdxl_ddim_select_editonly4_full700_tpami_v1": {
        "selection_policy": "score",
        "selection_clip_weight": 1.0,
        "selection_preserve_weight": 0.0,
        "selection_structure_weight": 0.0,
        "selection_min_preserve_wins_for_edit": 0,
        "selection_edit_clip_fraction": 1.0,
    },
}

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
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--python", default=os.environ.get("PYTHON", "python"))
    parser.add_argument("--hf_home", default=os.environ.get("HF_HOME", ""))
    parser.add_argument("--annotation_mapping_file", required=True)
    parser.add_argument("--annotation_image_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--result_path", required=True)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected", type=int, default=700)
    parser.add_argument("--skip_selectors", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    return parser.parse_args()


def run(cmd, args, name):
    env = os.environ.copy()
    if args.hf_home:
        env["HF_HOME"] = args.hf_home
    if args.device.startswith("cuda"):
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    start = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            cmd,
            cwd=args.repo,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.time() - start
    print(f"{name} exit={proc.returncode} elapsed={elapsed:.1f}s log={log_path}", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed; see {log_path}")


def selector_cmd(args, case_key, options):
    schedules = [
        ("trdi", CASE_TRDI),
        ("adaptive_noise_floor50", CASE_FLOOR50),
        ("adaptive_noise_floor75", CASE_FLOOR75),
        ("adaptive_late", CASE_LATE),
    ]
    roots = ",".join(
        f"{schedule}={Path(args.output_root) / source_case}"
        for schedule, source_case in schedules
    )
    cmd = [
        args.python,
        "scripts/run_icml_main_case_from_scratch.py",
        "--case_key",
        case_key,
        "--method",
        "ddim",
        "--model_type",
        "SDXL",
        "--num_inference_steps",
        "50",
        "--spacing",
        "1.05",
        "--trdi_window",
        "8",
        "--schedule_mode",
        "trdi",
        "--candidate_schedule_modes",
        ",".join(schedule for schedule, _ in schedules),
        "--candidate_image_roots",
        roots,
        "--candidate_selection_mode",
        "masked",
        "--selection_clip_weight",
        str(options.get("selection_clip_weight", 1.0)),
        "--selection_preserve_weight",
        str(options.get("selection_preserve_weight", 0.4)),
        "--selection_structure_weight",
        str(options.get("selection_structure_weight", 0.2)),
        "--selection_edit_clip_fraction",
        str(options.get("selection_edit_clip_fraction", 1.0)),
        "--selection_policy",
        options.get("selection_policy", "dominance"),
        "--selection_target_clip_drop_tolerance",
        str(options.get("selection_target_clip_drop_tolerance", 0.0)),
        "--selection_edit_clip_drop_tolerance",
        str(options.get("selection_edit_clip_drop_tolerance", 0.0)),
        "--selection_min_preserve_wins",
        str(options.get("selection_min_preserve_wins", 5)),
        "--selection_min_preserve_wins_for_edit",
        str(options.get("selection_min_preserve_wins_for_edit", 3)),
        "--guidance_scale",
        "1.0",
        "--annotation_mapping_file",
        args.annotation_mapping_file,
        "--annotation_image_root",
        args.annotation_image_root,
        "--output_root",
        args.output_root,
        "--start_index",
        "0",
        "--max_samples",
        str(args.expected),
        "--device",
        args.device,
    ]
    return cmd


def evaluate_cmd(args, methods):
    return [
        args.python,
        "evaluate_my.py",
        "--annotation_mapping_file",
        args.annotation_mapping_file,
        "--src_image_folder",
        args.annotation_image_root,
        "--tgt_path",
        args.output_root,
        "--tgt_methods",
        *methods,
        "--result_path",
        args.result_path,
        "--annotation_images",
        "--edit_category_list",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "--device",
        args.device,
    ]


def float_or_nan(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def nanmean(values):
    clean = [value for value in values if not math.isnan(value)]
    return sum(clean) / len(clean) if clean else math.nan


def summarize_method(result_path, method):
    csv_path = Path(result_path) / f"{method}.csv"
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    data = [row for row in rows if row.get("file_id") != "Avg"]
    return {
        metric: nanmean([float_or_nan(row.get(metric)) for row in data])
        for metric, _ in METRICS
    }


def summarize(args, methods):
    summaries = {method: summarize_method(args.result_path, method) for method in [CASE_TRDI, *methods]}
    baseline = summaries[CASE_TRDI]
    comparison = {}
    for method in methods:
        wins = 0
        metric_delta = {}
        for metric, higher_is_better in METRICS:
            delta = summaries[method][metric] - baseline[metric]
            good = delta > 0 if higher_is_better else delta < 0
            wins += int(good)
            metric_delta[metric] = {
                "baseline": baseline[metric],
                "method": summaries[method][metric],
                "delta": delta,
                "good": good,
            }
        comparison[method] = {"wins": wins, "metrics": metric_delta}
    output_path = Path(args.result_path) / "tpami_selector_ablation_summary.json"
    output_path.write_text(
        json.dumps({"methods": summaries, "comparison_to_trdi": comparison}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(comparison, indent=2), flush=True)
    print(f"summary={output_path}", flush=True)


def main():
    args = parse_args()
    methods = list(ABLATIONS)
    if not args.skip_selectors:
        for method, options in ABLATIONS.items():
            run(selector_cmd(args, method, options), args, method)
    if not args.skip_eval:
        run(evaluate_cmd(args, methods), args, "evaluate_selector_ablations")
        summarize(args, methods)


if __name__ == "__main__":
    main()
