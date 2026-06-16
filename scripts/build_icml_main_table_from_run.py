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
    "sdxl_ddim_base": ("SDXL", "DDIM", "Baseline"),
    "sdxl_ddim_trdi": ("SDXL", "DDIM", "w/ Ours"),
    "sdxl_renoise_base": ("SDXL", "ReNoise", "Baseline"),
    "sdxl_renoise_trdi": ("SDXL", "ReNoise", "w/ Ours"),
    "sdxl_npi_base": ("SDXL", "NPI", "Baseline"),
    "sdxl_npi_trdi": ("SDXL", "NPI", "w/ Ours"),
    "sdxl_gnri_base": ("SDXL", "GNRI", "Baseline"),
    "sdxl_gnri_trdi": ("SDXL", "GNRI", "w/ Ours"),
    "sdxlturbo_ddim_base": ("SDXL Turbo", "DDIM", "Baseline"),
    "sdxlturbo_ddim_trdi": ("SDXL Turbo", "DDIM", "w/ Ours"),
    "sdxlturbo_renoise_base": ("SDXL Turbo", "ReNoise", "Baseline"),
    "sdxlturbo_renoise_trdi": ("SDXL Turbo", "ReNoise", "w/ Ours"),
    "sdxlturbo_npi_base": ("SDXL Turbo", "NPI", "Baseline"),
    "sdxlturbo_npi_trdi": ("SDXL Turbo", "NPI", "w/ Ours"),
    "sdxlturbo_gnri_base": ("SDXL Turbo", "GNRI", "Baseline"),
    "sdxlturbo_gnri_trdi": ("SDXL Turbo", "GNRI", "w/ Ours"),
}


def clean_method(cell):
    return re.sub(r"\s+", " ", re.sub(r"\\[a-zA-Z]+|\{|\}", "", cell)).strip()


def first_float(cell):
    match = re.search(r"-?\d+(?:\.\d+)?", cell)
    return float(match.group(0)) if match else math.nan


def parse_paper_table(path):
    rows = []
    current_model = None
    current_method = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if "&" not in line or "\\\\" not in line:
            continue
        if "Distance" in line or "Model" in line or "cmidrule" in line:
            continue
        cells = [cell.strip() for cell in line.rsplit("\\\\", 1)[0].split("&")]
        if len(cells) < 9:
            continue
        if "SDXL" in cells[0]:
            current_model = "SDXL Turbo" if "Turbo" in cells[0] else "SDXL"
        method = clean_method(cells[1])
        if not method:
            continue
        variant = "w/ Ours" if "Ours" in method else "Baseline"
        if variant == "Baseline":
            current_method = method
        if current_model is None or current_method is None:
            continue
        values = {metric: first_float(cell) for (metric, _), cell in zip(METRICS, cells[2:])}
        rows.append({"model": current_model, "method": current_method, "variant": variant, **values})
    return rows


def read_run_csv(path):
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
        return {"complete": False, "n_samples": n_samples, "values": None}
    values = {}
    for metric, csv_key in CSV_KEYS.items():
        raw = avg.get(csv_key)
        values[metric] = float(raw) * SCALE[metric] if raw not in (None, "", "nan") else math.nan
    return {"complete": True, "n_samples": n_samples, "values": values}


def collect_run_rows(eval_dir):
    rows = {}
    for case_key, table_key in CASE_ROWS.items():
        run = read_run_csv(eval_dir / f"{case_key}.csv")
        if run and run["complete"]:
            rows[table_key] = run
    return rows


def fmt(value):
    if value is None or math.isnan(value):
        return "NA"
    return f"{value:.2f}"


def write_outputs(paper_rows, run_rows, eval_dir, output_md, output_csv):
    headers = ["Model", "Method", "Variant", "Source", "N"] + [name for _, name in METRICS]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in paper_rows:
            key = (row["model"], row["method"], row["variant"])
            writer.writerow([row["model"], row["method"], row["variant"], "paper", "", *[fmt(row[m]) for m, _ in METRICS]])
            run = run_rows.get(key)
            if run:
                writer.writerow(
                    [
                        row["model"],
                        row["method"],
                        row["variant"],
                        "run",
                        run["n_samples"],
                        *[fmt(run["values"].get(m, math.nan)) for m, _ in METRICS],
                    ]
                )

    lines = [
        "# ICML Main Table From Fresh Run",
        "",
        f"Evaluation directory: `{eval_dir}`.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in paper_rows:
        key = (row["model"], row["method"], row["variant"])
        lines.append(
            "| "
            + " | ".join([row["model"], row["method"], row["variant"], "paper", "", *[fmt(row[m]) for m, _ in METRICS]])
            + " |"
        )
        run = run_rows.get(key)
        if run:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["model"],
                        row["method"],
                        row["variant"],
                        "run",
                        str(run["n_samples"]),
                        *[fmt(run["values"].get(m, math.nan)) for m, _ in METRICS],
                    ]
                )
                + " |"
            )
    missing = [
        (row["model"], row["method"], row["variant"])
        for row in paper_rows
        if (row["model"], row["method"], row["variant"]) not in run_rows
    ]
    lines.extend(["", "## Pending Rows", ""])
    if missing:
        for model, method, variant in missing:
            lines.append(f"- {model} / {method} / {variant}")
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Run rows are read only from the supplied fresh-run `evaluation/` directory.",
            "- Metric scaling matches the paper table: Structure x1e3, LPIPS x1e3, MSE x1e4, SSIM x1e2.",
        ]
    )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_table", required=True, help="Path to the LaTeX table from the paper/source package.")
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    paper_rows = parse_paper_table(Path(args.paper_table))
    run_rows = collect_run_rows(Path(args.eval_dir))
    output_md = Path(args.output_md)
    output_csv = Path(args.output_csv)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_outputs(paper_rows, run_rows, Path(args.eval_dir), output_md, output_csv)
    print(f"read {len(run_rows)}/{len(CASE_ROWS)} completed run rows")
    print(f"wrote {output_md}")
    print(f"wrote {output_csv}")


if __name__ == "__main__":
    main()
