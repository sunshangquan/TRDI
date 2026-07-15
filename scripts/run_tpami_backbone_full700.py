#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import subprocess
import time
from pathlib import Path


SCHEDULES = [
    ("trdi", "trdi"),
    ("adaptive_noise_floor50", "adaptive_noise_floor50"),
    ("adaptive_noise_floor75", "adaptive_noise_floor75"),
    ("adaptive_late", "adaptive_late"),
]

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
    parser.add_argument("--method", required=True, choices=["ddim", "npi", "gnri", "renoise"])
    parser.add_argument("--model_type", default="SDXL", choices=["SD15", "SD21", "SDXL", "SDXL_Turbo"])
    parser.add_argument("--case_prefix", required=True)
    parser.add_argument("--case_tag", default="full700")
    parser.add_argument("--annotation_mapping_file", required=True)
    parser.add_argument("--annotation_image_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--result_path", required=True)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--expected", type=int, default=700)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--spacing", type=float, default=1.05)
    parser.add_argument("--trdi_window", type=int, default=8)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=700)
    parser.add_argument("--skip_candidates", action="store_true")
    parser.add_argument("--skip_selector", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--selector_gates", default="balanced,conservative")
    parser.add_argument("--poll_seconds", type=int, default=60)
    parser.add_argument("--renoise_steps", type=int, default=1)
    parser.add_argument("--early_timestep", type=int, default=250)
    parser.add_argument("--early_weights", default="0.5,0.5")
    parser.add_argument("--latter_weights", default="0.0,1.0")
    parser.add_argument("--lambda_pair", type=int, default=10)
    parser.add_argument("--lambda_patch_kl", type=float, default=0.05)
    parser.add_argument("--num_reg_steps", type=int, default=4)
    parser.add_argument("--num_ac_rolls", type=int, default=5)
    parser.add_argument("--num_iter_steps", type=int, default=2)
    parser.add_argument("--param_lambda", type=float, default=0.1)
    parser.add_argument("--param_eta", type=float, default=0.0)
    return parser.parse_args()


def case_key(args, schedule_mode):
    return f"{args.case_prefix}_{schedule_mode}_{args.case_tag}_tpami_v1"


def selector_case_key(args, gate):
    if gate == "balanced":
        return f"{args.case_prefix}_select_balanced4_{args.case_tag}_tpami_v1"
    if gate == "conservative":
        return f"{args.case_prefix}_select_strictclip4_{args.case_tag}_tpami_v1"
    if gate == "safe":
        return f"{args.case_prefix}_select_safe4sm_{args.case_tag}_tpami_v1"
    raise ValueError(f"unknown gate: {gate}")


def run_env(args, device):
    env = os.environ.copy()
    if args.hf_home:
        env["HF_HOME"] = args.hf_home
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    return env


def candidate_cmd(args, schedule_mode):
    cmd = [
        args.python,
        "scripts/run_icml_main_case_from_scratch.py",
        "--case_key",
        case_key(args, schedule_mode),
        "--method",
        args.method,
        "--model_type",
        args.model_type,
        "--num_inference_steps",
        str(args.num_inference_steps),
        "--spacing",
        str(args.spacing),
        "--trdi_window",
        str(args.trdi_window),
        "--schedule_mode",
        schedule_mode,
        "--guidance_scale",
        str(args.guidance_scale),
        "--annotation_mapping_file",
        args.annotation_mapping_file,
        "--annotation_image_root",
        args.annotation_image_root,
        "--output_root",
        args.output_root,
        "--start_index",
        str(args.start_index),
        "--max_samples",
        str(args.max_samples),
        "--device",
        "cuda",
        "--renoise_steps",
        str(args.renoise_steps),
        "--early_timestep",
        str(args.early_timestep),
        "--early_weights",
        args.early_weights,
        "--latter_weights",
        args.latter_weights,
        "--lambda_pair",
        str(args.lambda_pair),
        "--lambda_patch_kl",
        str(args.lambda_patch_kl),
        "--num_reg_steps",
        str(args.num_reg_steps),
        "--num_ac_rolls",
        str(args.num_ac_rolls),
        "--num_iter_steps",
        str(args.num_iter_steps),
        "--param_lambda",
        str(args.param_lambda),
        "--param_eta",
        str(args.param_eta),
    ]
    return cmd


def selector_cmd(args, gate):
    roots = ",".join(
        f"{schedule_mode}={Path(args.output_root) / case_key(args, schedule_mode)}"
        for schedule_mode, _ in SCHEDULES
    )
    min_preserve_wins_for_edit = "2" if gate == "balanced" else "3"
    cmd = [
        args.python,
        "scripts/run_icml_main_case_from_scratch.py",
        "--case_key",
        selector_case_key(args, gate),
        "--method",
        args.method,
        "--model_type",
        args.model_type,
        "--num_inference_steps",
        str(args.num_inference_steps),
        "--spacing",
        str(args.spacing),
        "--trdi_window",
        str(args.trdi_window),
        "--schedule_mode",
        "trdi",
        "--candidate_schedule_modes",
        ",".join(schedule_mode for schedule_mode, _ in SCHEDULES),
        "--candidate_image_roots",
        roots,
        "--candidate_selection_mode",
        "masked",
        "--selection_policy",
        "dominance",
        "--selection_clip_weight",
        "1.0",
        "--selection_preserve_weight",
        "0.4",
        "--selection_structure_weight",
        "0.2",
        "--selection_edit_clip_fraction",
        "1.0",
        "--selection_target_clip_drop_tolerance",
        "0.0",
        "--selection_edit_clip_drop_tolerance",
        "0.0",
        "--selection_min_preserve_wins",
        "5",
        "--selection_min_preserve_wins_for_edit",
        min_preserve_wins_for_edit,
        "--guidance_scale",
        str(args.guidance_scale),
        "--annotation_mapping_file",
        args.annotation_mapping_file,
        "--annotation_image_root",
        args.annotation_image_root,
        "--output_root",
        args.output_root,
        "--start_index",
        str(args.start_index),
        "--max_samples",
        str(args.max_samples),
        "--device",
        "cuda",
    ]
    if gate == "safe":
        cmd.extend(
            [
                "--selection_max_structure_increase",
                "0.0",
                "--selection_max_mse_increase",
                "0.0",
            ]
        )
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
        "cuda",
    ]


def count_images(output_root, case):
    case_dir = Path(output_root) / case
    if not case_dir.is_dir():
        return 0
    return sum(1 for _ in case_dir.rglob("*.jpg"))


def read_summary(output_root, case):
    path = Path(output_root) / case / "run_summary.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def launch_candidates(args):
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices:
        raise ValueError("--devices must contain at least one CUDA device id")
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    queue = list(SCHEDULES)
    procs = []

    def launch_one(schedule_mode, device):
        name = case_key(args, schedule_mode)
        log_path = log_dir / f"{name}.log"
        print(f"launch {name} on cuda:{device} log={log_path}", flush=True)
        handle = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            candidate_cmd(args, schedule_mode),
            cwd=args.repo,
            env=run_env(args, device),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return name, proc, handle, device

    for device in devices:
        if not queue:
            break
        schedule_mode, _ = queue.pop(0)
        procs.append(launch_one(schedule_mode, device))
    failed = []
    while procs or queue:
        remaining = []
        statuses = []
        free_devices = []
        for name, proc, handle, device in procs:
            code = proc.poll()
            count = count_images(args.output_root, name)
            statuses.append(f"{name}:{count}/{args.expected}{' running' if code is None else ' exit='+str(code)}")
            if code is None:
                remaining.append((name, proc, handle, device))
            else:
                handle.close()
                free_devices.append(device)
                if code != 0:
                    failed.append((name, code))
        print("candidate status " + " | ".join(statuses), flush=True)
        if failed:
            raise RuntimeError(f"candidate failures: {failed}")
        for device in free_devices:
            if not queue:
                break
            schedule_mode, _ = queue.pop(0)
            remaining.append(launch_one(schedule_mode, device))
        procs = remaining
        if procs:
            time.sleep(args.poll_seconds)


def wait_candidates(args, methods):
    while True:
        complete = True
        statuses = []
        for method in methods:
            summary = read_summary(args.output_root, method)
            count = count_images(args.output_root, method)
            failures = summary.get("failures", []) if summary else []
            done = summary is not None and summary.get("completed", 0) + summary.get("skipped", 0) >= args.expected
            done = done and count >= args.expected and not failures
            statuses.append(f"{method}:{count}/{args.expected}{' done' if done else ''}")
            complete = complete and done
            if failures:
                raise RuntimeError(f"{method} failures: {failures}")
        print("candidate status " + " | ".join(statuses), flush=True)
        if complete:
            return
        time.sleep(args.poll_seconds)


def run_blocking(cmd, args, name, device):
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    start = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            cmd,
            cwd=args.repo,
            env=run_env(args, device),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.time() - start
    print(f"{name} exit={proc.returncode} elapsed={elapsed:.1f}s log={log_path}", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed; see {log_path}")


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
    summaries = {method: summarize_method(args.result_path, method) for method in methods}
    baseline = summaries[case_key(args, "trdi")]
    comparison = {}
    for method in methods:
        if method == case_key(args, "trdi"):
            continue
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
    output_path = Path(args.result_path) / f"{args.case_prefix}_{args.case_tag}_summary.json"
    output_path.write_text(
        json.dumps({"methods": summaries, "comparison_to_trdi": comparison}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(comparison, indent=2), flush=True)
    print(f"summary={output_path}", flush=True)


def main():
    args = parse_args()
    candidate_methods = [case_key(args, schedule_mode) for schedule_mode, _ in SCHEDULES]
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    selector_device = devices[-1] if devices else "0"
    gates = [item.strip() for item in args.selector_gates.split(",") if item.strip()]
    invalid_gates = sorted(set(gates) - {"balanced", "conservative", "safe"})
    if invalid_gates:
        raise ValueError(f"invalid selector gates: {invalid_gates}")
    if not args.skip_candidates:
        launch_candidates(args)
    else:
        wait_candidates(args, candidate_methods)
    selectors = [selector_case_key(args, gate) for gate in gates]
    if not args.skip_selector:
        for gate, selector in zip(gates, selectors):
            run_blocking(selector_cmd(args, gate), args, selector, selector_device)
    methods = [*candidate_methods, *selectors]
    if not args.skip_eval:
        run_blocking(evaluate_cmd(args, methods), args, f"evaluate_{args.case_prefix}_{args.case_tag}", selector_device)
        summarize(args, methods)


if __name__ == "__main__":
    main()
