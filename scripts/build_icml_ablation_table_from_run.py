#!/usr/bin/env python3

import argparse
import csv
import math
import re
from pathlib import Path


METRICS = [
    ("structure", "Structure x1e3"),
    ("psnr", "PSNR"),
    ("lpips", "LPIPS x1e3"),
    ("mse", "MSE x1e4"),
    ("ssim", "SSIM x1e2"),
    ("clip_whole", "Whole CLIP"),
    ("clip_edited", "Edited CLIP"),
]

CSV_KEYS = {
    "structure": "structure_distance",
    "psnr": "psnr_unedit_part",
    "lpips": "lpips_unedit_part",
    "mse": "mse_unedit_part",
    "ssim": "ssim_unedit_part",
    "clip_whole": "clip_similarity_target_image",
    "clip_edited": "clip_similarity_target_image_edit_part",
}

SCALE = {
    "structure": 1000.0,
    "psnr": 1.0,
    "lpips": 1000.0,
    "mse": 10000.0,
    "ssim": 100.0,
    "clip_whole": 1.0,
    "clip_edited": 1.0,
}

CASE_ROWS = {
    ("1.10", "0"): "sdxl_ddim_gamma_1p10_d0",
    ("1.05", "0"): "sdxl_ddim_gamma_1p05_d0",
    ("1", "0"): "sdxl_ddim_base",
    ("0.90", "0"): "sdxl_ddim_gamma_0p90_d0",
    ("1.05", "2"): "sdxl_ddim_gamma_1p05_d2",
    ("1.05", "5"): "sdxl_ddim_gamma_1p05_d5",
    ("1.05", "8"): "sdxl_ddim_trdi",
    ("1.05", "10"): "sdxl_ddim_gamma_1p05_d10",
}


def first_float(cell):
    match = re.search(r"-?\d+(?:\.\d+)?", cell)
    return float(match.group(0)) if match else math.nan


def normalize_setting(cell):
    value = first_float(cell)
    if math.isnan(value):
        return ""
    if abs(value - round(value)) < 1e-8:
        return str(int(round(value)))
    return f"{value:.2f}"


def parse_paper_table(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "&" not in line or "\\\\" not in line:
            continue
        if "Settings" in line or "\\gamma" in line or "Distance" in line or "midrule" in line:
            continue
        cells = [cell.strip() for cell in line.rsplit("\\\\", 1)[0].split("&")]
        if len(cells) < 9:
            continue
        gamma = normalize_setting(cells[0])
        window = normalize_setting(cells[1])
        if not gamma or not window:
            continue
        values = {metric: first_float(cell) for (metric, _), cell in zip(METRICS, cells[2:])}
        rows.append({"gamma": gamma, "window": window, **values})
    return rows


def read_avg(path):
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    avg = None
    n_samples = 0
    for row in rows:
        if row.get("file_id") == "Avg":
            avg = row
        else:
            n_samples += 1
    if avg is None:
        return None
    values = {}
    for metric, csv_key in CSV_KEYS.items():
        raw = avg.get(csv_key)
        values[metric] = float(raw) * SCALE[metric] if raw not in (None, "", "nan") else math.nan
    return {"n_samples": n_samples, "values": values}


def fmt(value):
    if value is None or math.isnan(value):
        return "NA"
    return f"{value:.2f}"


def write_outputs(paper_rows, eval_dir, output_md, output_csv):
    headers = ["gamma", "d", "Source", "N"] + [name for _, name in METRICS]
    run_rows = {}
    for settings, case_key in CASE_ROWS.items():
        run = read_avg(eval_dir / f"{case_key}.csv")
        if run:
            run_rows[settings] = run

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in paper_rows:
            key = (row["gamma"], row["window"])
            writer.writerow([row["gamma"], row["window"], "paper", "", *[fmt(row[m]) for m, _ in METRICS]])
            run = run_rows.get(key)
            if run:
                writer.writerow(
                    [
                        row["gamma"],
                        row["window"],
                        "run",
                        run["n_samples"],
                        *[fmt(run["values"].get(m, math.nan)) for m, _ in METRICS],
                    ]
                )

    lines = [
        "# ICML SDXL DDIM Ablation Table From Fresh Run",
        "",
        f"Evaluation directory: `{eval_dir}`.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in paper_rows:
        key = (row["gamma"], row["window"])
        lines.append("| " + " | ".join([row["gamma"], row["window"], "paper", "", *[fmt(row[m]) for m, _ in METRICS]]) + " |")
        run = run_rows.get(key)
        if run:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["gamma"],
                        row["window"],
                        "run",
                        str(run["n_samples"]),
                        *[fmt(run["values"].get(m, math.nan)) for m, _ in METRICS],
                    ]
                )
                + " |"
            )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(run_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_table", required=True, help="Path to the LaTeX SDXL DDIM ablation table.")
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()
    paper_rows = parse_paper_table(Path(args.paper_table))
    output_md = Path(args.output_md)
    output_csv = Path(args.output_csv)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    n_rows = write_outputs(paper_rows, Path(args.eval_dir), output_md, output_csv)
    print(f"read {n_rows}/{len(CASE_ROWS)} completed run rows")
    print(f"wrote {output_md}")
    print(f"wrote {output_csv}")


if __name__ == "__main__":
    main()
