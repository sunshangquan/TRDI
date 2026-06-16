#!/usr/bin/env python3

import argparse
import csv
import math
import re
from pathlib import Path


METRICS = [
    ("psnr", "PSNR"),
    ("ssim", "SSIM x1e2"),
    ("lpips", "LPIPS x1e3"),
]

SCALE = {"psnr": 1.0, "ssim": 100.0, "lpips": 1000.0}

CASE_ROWS = {
    "sd15_ddim_base": ("DDIM", "Baseline"),
    "sd15_ddim_trdi": ("DDIM", "w/ Ours"),
    "sd15_renoise_base": ("ReNoise", "Baseline"),
    "sd15_renoise_trdi": ("ReNoise", "w/ Ours"),
    "sd15_npi_base": ("NPI", "Baseline"),
    "sd15_npi_trdi": ("NPI", "w/ Ours"),
    "sd15_gnri_base": ("GNRI", "Baseline"),
    "sd15_gnri_trdi": ("GNRI", "w/ Ours"),
}


def clean_method(cell):
    return re.sub(r"\s+", " ", re.sub(r"\\[a-zA-Z]+|\{|\}", "", cell)).strip()


def first_float(cell):
    match = re.search(r"-?\d+(?:\.\d+)?", cell)
    return float(match.group(0)) if match else math.nan


def parse_paper_table(path):
    rows = []
    current_method = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if "&" not in line or "\\\\" not in line:
            continue
        if "PSNR" in line or "Method" in line or "midrule" in line:
            continue
        cells = [cell.strip() for cell in line.rsplit("\\\\", 1)[0].split("&")]
        if len(cells) < 4:
            continue
        method = clean_method(cells[0])
        if not method:
            continue
        variant = "w/ Ours" if "Ours" in method else "Baseline"
        if variant == "Baseline":
            current_method = method
        if current_method is None:
            continue
        values = {metric: first_float(cell) for (metric, _), cell in zip(METRICS, cells[1:])}
        rows.append({"method": current_method, "variant": variant, **values})
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
    for metric, _ in METRICS:
        raw = avg.get(metric)
        values[metric] = float(raw) * SCALE[metric] if raw not in (None, "", "nan") else math.nan
    return {"n_samples": n_samples, "values": values}


def collect_run_rows(eval_dir):
    rows = {}
    for case_key, table_key in CASE_ROWS.items():
        run = read_avg(eval_dir / f"{case_key}.csv")
        if run:
            rows[table_key] = run
    return rows


def fmt(value):
    if value is None or math.isnan(value):
        return "NA"
    return f"{value:.2f}"


def write_outputs(paper_rows, run_rows, eval_dir, output_md, output_csv):
    headers = ["Method", "Variant", "Source", "N"] + [name for _, name in METRICS]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in paper_rows:
            key = (row["method"], row["variant"])
            writer.writerow([row["method"], row["variant"], "paper", "", *[fmt(row[m]) for m, _ in METRICS]])
            run = run_rows.get(key)
            if run:
                writer.writerow(
                    [
                        row["method"],
                        row["variant"],
                        "run",
                        run["n_samples"],
                        *[fmt(run["values"].get(m, math.nan)) for m, _ in METRICS],
                    ]
                )

    lines = [
        "# ICML Reconstruction Table From Fresh Run",
        "",
        f"Evaluation directory: `{eval_dir}`.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in paper_rows:
        key = (row["method"], row["variant"])
        lines.append("| " + " | ".join([row["method"], row["variant"], "paper", "", *[fmt(row[m]) for m, _ in METRICS]]) + " |")
        run = run_rows.get(key)
        if run:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["method"],
                        row["variant"],
                        "run",
                        str(run["n_samples"]),
                        *[fmt(run["values"].get(m, math.nan)) for m, _ in METRICS],
                    ]
                )
                + " |"
            )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_table", required=True, help="Path to the LaTeX reconstruction table.")
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
