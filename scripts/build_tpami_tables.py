#!/usr/bin/env python3

import argparse
import csv
import json
import math
import re
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = ROOT / "final_reports" / "tpami_sources"
DEFAULT_OUTPUT_DIR = ROOT / "final_reports"
SOURCE_DIR = DEFAULT_SOURCE_DIR
OUTPUT_DIR = DEFAULT_OUTPUT_DIR

METRICS = [
    ("structure_distance", "Struct. $\\downarrow$", 6),
    ("psnr_unedit_part", "PSNR $\\uparrow$", 4),
    ("lpips_unedit_part", "LPIPS $\\downarrow$", 6),
    ("mse_unedit_part", "MSE $\\downarrow$", 6),
    ("ssim_unedit_part", "SSIM $\\uparrow$", 6),
    ("clip_similarity_target_image", "CLIP-T $\\uparrow$", 4),
    ("clip_similarity_target_image_edit_part", "CLIP-E $\\uparrow$", 4),
]

DISPLAY_NAMES = {
    "trdi": "TRDI",
    "npi_trdi": "NPI-TRDI",
    "gnri_trdi": "GNRI-TRDI",
    "renoise_trdi": "ReNoise-TRDI",
    "adaptive_noise_floor50": "Adaptive-F50",
    "adaptive_noise_floor75": "Adaptive-F75",
    "adaptive_late": "Adaptive-Late",
    "balanced": "CGA-TRDI-Balanced",
    "conservative": "CGA-TRDI-Conservative",
    "safe": "CGA-TRDI-Safe",
    "score": "Score-only",
    "editonly": "Edit-only",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_dir",
        default=str(DEFAULT_SOURCE_DIR),
        help="Directory containing TPAMI summary JSON files and record/run-summary subfolders.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where CSV, Markdown, LaTeX, and XLSX tables will be written.",
    )
    return parser.parse_args()


def read_json(name):
    with (SOURCE_DIR / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def has_source(name):
    return (SOURCE_DIR / name).is_file()


def method_label(method):
    if "_select_safe4sm_" in method:
        return DISPLAY_NAMES["safe"]
    if "_select_strictclip4_" in method:
        return DISPLAY_NAMES["conservative"]
    if "_select_relaxed2_" in method or "_select_balanced4_" in method:
        return DISPLAY_NAMES["balanced"]
    if "_select_score4_" in method:
        return DISPLAY_NAMES["score"]
    if "_select_editonly4_" in method:
        return DISPLAY_NAMES["editonly"]
    if "_adaptive_noise_floor50_" in method:
        return DISPLAY_NAMES["adaptive_noise_floor50"]
    if "_adaptive_noise_floor75_" in method:
        return DISPLAY_NAMES["adaptive_noise_floor75"]
    if "_adaptive_late_" in method:
        return DISPLAY_NAMES["adaptive_late"]
    if method.startswith("sdxl_npi"):
        return DISPLAY_NAMES["npi_trdi"]
    if method.startswith("sdxl_gnri"):
        return DISPLAY_NAMES["gnri_trdi"]
    if method.startswith("sdxl_renoise"):
        return DISPLAY_NAMES["renoise_trdi"]
    return DISPLAY_NAMES["trdi"]


def fmt(value, places):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    return f"{float(value):.{places}f}"


def aggregate_wins(summary, method):
    baseline = summary["baseline"]
    if method == baseline:
        return "-"
    return f"{summary['comparison_to_baseline'][method]['wins']}/7"


def category_wins(summary, method):
    baseline = summary["baseline"]
    if method == baseline:
        return "-"
    category = summary["category"][method]
    return f"{category['category_metric_wins']}/{category['category_metric_total']}"


def category_metric_wins(summary, method, category):
    if method == summary["baseline"]:
        return "-"
    category_summary = summary["category"][method]["categories"][str(category)]
    return f"{category_summary['wins']}/7"


def switched_count(summary, method):
    counts = summary.get("selection_counts", {}).get(method)
    if not counts:
        return "-"
    total = counts["total"]
    trdi = counts["counts"].get("trdi", 0)
    return f"{total - trdi}/{total}"


def schedule_counts_from_records(record_path):
    path = SOURCE_DIR / record_path
    records = json.loads(path.read_text(encoding="utf-8"))
    out = {str(category): {"trdi": 0, "adaptive_noise_floor50": 0, "adaptive_noise_floor75": 0, "adaptive_late": 0} for category in range(10)}
    for record in records:
        category = Path(record["file"]).parts[0].split("_", 1)[0]
        selected = record["selected_schedule_mode"]
        if category not in out:
            out[category] = {"trdi": 0, "adaptive_noise_floor50": 0, "adaptive_noise_floor75": 0, "adaptive_late": 0}
        out[category][selected] = out[category].get(selected, 0) + 1
    return out


def metric_row(summary, method, group=None):
    aggregate = summary["aggregate"][method]
    row = []
    if group is not None:
        row.append(group)
    row.append(method_label(method))
    row.extend(fmt(aggregate[key], places) for key, _, places in METRICS)
    row.append(aggregate_wins(summary, method))
    row.append(category_wins(summary, method))
    row.append(switched_count(summary, method))
    return row


def main_table_rows():
    ddim = read_json("ddim_full700_summary.json")
    ddim_methods = [
        "sdxl_ddim_trdi_full700_tpami_v1",
        "sdxl_ddim_adaptive_noise_floor50_full700_tpami_v1",
        "sdxl_ddim_adaptive_noise_floor75_full700_tpami_v1",
        "sdxl_ddim_adaptive_late_full700_tpami_v1",
        "sdxl_ddim_select_relaxed2_full700_tpami_v1",
        "sdxl_ddim_select_strictclip4_full700_tpami_v1",
    ]
    header = [
        "Method",
        *[label for _, label, _ in METRICS],
        "Agg. wins",
        "Category wins",
        "Switched",
    ]
    rows = [header]
    rows.extend(metric_row(ddim, method) for method in ddim_methods)
    return rows


def ddim_category_selection_rows():
    summary = read_json("ddim_full700_summary.json")
    balanced = "sdxl_ddim_select_relaxed2_full700_tpami_v1"
    conservative = "sdxl_ddim_select_strictclip4_full700_tpami_v1"
    balanced_counts = schedule_counts_from_records("records/ddim_balanced/selection_records.json")
    conservative_counts = schedule_counts_from_records("records/ddim_conservative/selection_records.json")
    header = [
        "Category",
        "Samples",
        "Balanced wins",
        "Balanced TRDI/F50/F75/Late",
        "Conservative wins",
        "Conservative TRDI/F50/F75/Late",
    ]
    rows = [header]
    for category in range(10):
        cat = str(category)
        samples = summary["category"][balanced]["categories"][cat]["samples"]
        b = balanced_counts[cat]
        c = conservative_counts[cat]
        rows.append(
            [
                cat,
                samples,
                category_metric_wins(summary, balanced, category),
                f"{b['trdi']}/{b['adaptive_noise_floor50']}/{b['adaptive_noise_floor75']}/{b['adaptive_late']}",
                category_metric_wins(summary, conservative, category),
                f"{c['trdi']}/{c['adaptive_noise_floor50']}/{c['adaptive_noise_floor75']}/{c['adaptive_late']}",
            ]
        )
    return rows


def npi_duplicate_rows():
    npi = read_json("npi_full700_summary.json")
    npi_methods = [
        "sdxl_npi_trdi_full700_tpami_v1",
        "sdxl_npi_adaptive_noise_floor50_full700_tpami_v1",
        "sdxl_npi_adaptive_noise_floor75_full700_tpami_v1",
        "sdxl_npi_adaptive_late_full700_tpami_v1",
        "sdxl_npi_select_balanced4_full700_tpami_v1",
        "sdxl_npi_select_strictclip4_full700_tpami_v1",
    ]
    header = [
        "Method",
        *[label for _, label, _ in METRICS],
        "Agg. wins",
        "Category wins",
        "Switched",
        "Use in paper",
    ]
    rows = [header]
    for method in npi_methods:
        row = metric_row(npi, method)
        row.append("sanity only")
        rows.append(row)
    return rows


def gnri_rows():
    summary = read_json("gnri_xcat10_summary.json")
    methods = [
        "sdxl_gnri_xcat10_trdi_xcat10_tpami_v1",
        "sdxl_gnri_xcat10_adaptive_noise_floor50_xcat10_tpami_v1",
        "sdxl_gnri_xcat10_adaptive_noise_floor75_xcat10_tpami_v1",
        "sdxl_gnri_xcat10_adaptive_late_xcat10_tpami_v1",
        "sdxl_gnri_xcat10_select_balanced4_xcat10_tpami_v1",
        "sdxl_gnri_xcat10_select_strictclip4_xcat10_tpami_v1",
        "sdxl_gnri_xcat10_select_safe4sm_xcat10_tpami_v1",
    ]
    header = [
        "Method",
        *[label for _, label, _ in METRICS],
        "Agg. wins",
        "Category wins",
        "Switched",
    ]
    return [header, *[metric_row(summary, method) for method in methods]]


def renoise_rows():
    summary = read_json("renoise_xcat10_summary.json")
    methods = [
        "sdxl_renoise_xcat10_trdi_xcat10_tpami_v1",
        "sdxl_renoise_xcat10_adaptive_noise_floor50_xcat10_tpami_v1",
        "sdxl_renoise_xcat10_adaptive_noise_floor75_xcat10_tpami_v1",
        "sdxl_renoise_xcat10_adaptive_late_xcat10_tpami_v1",
        "sdxl_renoise_xcat10_select_balanced4_xcat10_tpami_v1",
        "sdxl_renoise_xcat10_select_strictclip4_xcat10_tpami_v1",
        "sdxl_renoise_xcat10_select_safe4sm_xcat10_tpami_v1",
    ]
    header = [
        "Method",
        *[label for _, label, _ in METRICS],
        "Agg. wins",
        "Category wins",
        "Switched",
    ]
    return [header, *[metric_row(summary, method) for method in methods]]


def ablation_rows():
    summary = read_json("ddim_selector_ablation_summary.json")
    full_summary = read_json("ddim_full700_summary.json")
    methods = [
        "sdxl_ddim_select_score4_full700_tpami_v1",
        "sdxl_ddim_select_editonly4_full700_tpami_v1",
        "sdxl_ddim_select_relaxed2_full700_tpami_v1",
        "sdxl_ddim_select_strictclip4_full700_tpami_v1",
    ]
    manual_category_wins = {
        "sdxl_ddim_select_score4_full700_tpami_v1": "42/70",
        "sdxl_ddim_select_editonly4_full700_tpami_v1": "37/70",
        "sdxl_ddim_select_relaxed2_full700_tpami_v1": "70/70",
        "sdxl_ddim_select_strictclip4_full700_tpami_v1": "70/70",
    }
    manual_switched = {
        "sdxl_ddim_select_score4_full700_tpami_v1": "467/700",
        "sdxl_ddim_select_editonly4_full700_tpami_v1": "458/700",
        "sdxl_ddim_select_relaxed2_full700_tpami_v1": "212/700",
        "sdxl_ddim_select_strictclip4_full700_tpami_v1": "179/700",
    }
    header = ["Variant", "Switched", "Agg. wins", "Category wins"]
    rows = [header]
    for method in methods:
        if method in summary["comparison_to_trdi"]:
            wins = summary["comparison_to_trdi"][method]["wins"]
        else:
            wins = full_summary["comparison_to_baseline"][method]["wins"]
        rows.append(
            [
                method_label(method),
                manual_switched[method],
                f"{wins}/7",
                manual_category_wins[method],
            ]
        )
    return rows


def read_run_summary(relative_path):
    path = SOURCE_DIR / relative_path
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def elapsed(relative_path):
    summary = read_run_summary(relative_path)
    return float(summary["elapsed_seconds"]) if summary else math.nan


def minutes(seconds):
    return fmt(seconds / 60.0, 2)


def gpu_hours(seconds):
    return fmt(seconds / 3600.0, 2)


def add_efficiency_setting(rows, setting, samples, base_prefix, selector_names):
    # Explicit paths keep this table tied to audited run_summary files.
    if base_prefix == "full700":
        prefix = "run_summaries/full700"
        timings = {
            "TRDI": elapsed(f"{prefix}/ddim_trdi_run_summary.json"),
            "Adaptive-F50": elapsed(f"{prefix}/ddim_f50_run_summary.json"),
            "Adaptive-F75": elapsed(f"{prefix}/ddim_f75_run_summary.json"),
            "Adaptive-Late": elapsed(f"{prefix}/ddim_late_run_summary.json"),
            "CGA-TRDI-Balanced selector": elapsed(f"{prefix}/ddim_balanced_selector_run_summary.json"),
            "CGA-TRDI-Conservative selector": elapsed(f"{prefix}/ddim_conservative_selector_run_summary.json"),
        }
    elif base_prefix == "renoise_xcat10":
        prefix = "run_summaries/renoise_xcat10"
        timings = {
            "TRDI": elapsed(f"{prefix}/renoise_trdi_run_summary.json"),
            "Adaptive-F50": elapsed(f"{prefix}/renoise_f50_run_summary.json"),
            "Adaptive-F75": elapsed(f"{prefix}/renoise_f75_run_summary.json"),
            "Adaptive-Late": elapsed(f"{prefix}/renoise_late_run_summary.json"),
            "CGA-TRDI-Balanced selector": elapsed(f"{prefix}/renoise_balanced_selector_run_summary.json"),
            "CGA-TRDI-Conservative selector": elapsed(f"{prefix}/renoise_conservative_selector_run_summary.json"),
            "CGA-TRDI-Safe selector": elapsed(f"{prefix}/renoise_safe_selector_run_summary.json"),
        }
    else:
        return

    trdi_seconds = timings["TRDI"]
    if not math.isfinite(trdi_seconds):
        return
    rows.append(
        [
            setting,
            "TRDI",
            samples,
            minutes(trdi_seconds),
            "0.00",
            minutes(trdi_seconds),
            gpu_hours(trdi_seconds),
            "1.00x",
        ]
    )
    candidate_seconds = [
        timings["TRDI"],
        timings["Adaptive-F50"],
        timings["Adaptive-F75"],
        timings["Adaptive-Late"],
    ]
    for selector_label in selector_names:
        selector_seconds = timings.get(f"{selector_label} selector")
        if selector_seconds is None or not math.isfinite(selector_seconds):
            continue
        candidate_wall = max(candidate_seconds)
        total_wall = candidate_wall + selector_seconds
        total_gpu_seconds = sum(candidate_seconds) + selector_seconds
        rows.append(
            [
                setting,
                selector_label,
                samples,
                minutes(candidate_wall),
                minutes(selector_seconds),
                minutes(total_wall),
                gpu_hours(total_gpu_seconds),
                f"{total_gpu_seconds / trdi_seconds:.2f}x",
            ]
        )


def efficiency_rows():
    header = [
        "Setting",
        "Variant",
        "Samples",
        "Candidate wall min",
        "Selector wall min",
        "Total wall min",
        "GPU-hours",
        "Relative GPU-hours",
    ]
    rows = [header]
    add_efficiency_setting(
        rows,
        "SDXL-DDIM full PIE-Bench",
        700,
        "full700",
        ["CGA-TRDI-Balanced", "CGA-TRDI-Conservative"],
    )
    add_efficiency_setting(
        rows,
        "SDXL-ReNoise xcat10",
        100,
        "renoise_xcat10",
        ["CGA-TRDI-Balanced", "CGA-TRDI-Conservative", "CGA-TRDI-Safe"],
    )
    return rows


def reconstruction_delta_rows():
    sources = [
        ("SD v1.5", OUTPUT_DIR / "recon_table_coco1000_curated.csv"),
        ("SDXL", OUTPUT_DIR / "recon_table_sdxl_coco1000_curated.csv"),
        ("SDXL-Turbo", OUTPUT_DIR / "recon_table_sdxlturbo_coco1000_curated.csv"),
    ]
    header = [
        "Model",
        "Method",
        "N",
        "PSNR delta",
        "SSIM x1e2 delta",
        "LPIPS x1e3 delta",
        "Paper use",
    ]
    rows = [header]
    for model, path in sources:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
        by_key = {
            (record["Method"], record["Variant"], record["Source"]): record
            for record in records
        }
        methods = sorted({record["Method"] for record in records})
        for method in methods:
            base = by_key.get((method, "Baseline", "run"))
            ours = by_key.get((method, "w/ Ours", "run"))
            if not base or not ours:
                continue
            psnr_delta = float(ours["PSNR"]) - float(base["PSNR"])
            ssim_delta = float(ours["SSIM x1e2"]) - float(base["SSIM x1e2"])
            lpips_delta = float(ours["LPIPS x1e3"]) - float(base["LPIPS x1e3"])
            rows.append(
                [
                    model,
                    method,
                    ours["N"],
                    fmt(psnr_delta, 4),
                    fmt(ssim_delta, 4),
                    fmt(lpips_delta, 4),
                    "within-run delta only",
                ]
            )
    return rows


def claim_evidence_rows():
    rows = [
        ["Claim", "Evidence", "Status"],
        [
            "Adaptive schedules expose useful alternatives but are not reliable alone.",
            "Static schedules are weak on full DDIM and GNRI xcat10, and ReNoise static schedules have only 35-39/70 category wins despite stronger aggregate preservation metrics.",
            "Supported",
        ],
        [
            "Confidence-gated selection is the core stabilizer.",
            "Balanced/Conservative reach 7/7 aggregate and 70/70 category wins on DDIM full PIE-Bench; score-only/edit-only ablations fail category stability.",
            "Supported",
        ],
        [
            "The method can be transferred beyond the main DDIM setting.",
            "GNRI xcat10 reaches 7/7 aggregate with the stricter Safe gate; ReNoise xcat10 reaches 7/7 aggregate with Balanced, Conservative, and Safe gates.",
            "Supported for cross-category probes",
        ],
        [
            "The Safe gate improves robustness under stress.",
            "GNRI xcat10 Safe switches 6/100 samples and turns the conservative 6/7 into 7/7 aggregate wins; ReNoise Safe keeps 7/7 aggregate wins while switching 18/100 samples.",
            "Supported as a conservative operating point",
        ],
    ]
    if has_source("renoise_xcat10_summary.json"):
        summary = read_json("renoise_xcat10_summary.json")
        safe = "sdxl_renoise_xcat10_select_safe4sm_xcat10_tpami_v1"
        conservative = "sdxl_renoise_xcat10_select_strictclip4_xcat10_tpami_v1"
        balanced = "sdxl_renoise_xcat10_select_balanced4_xcat10_tpami_v1"
        parts = []
        for name, method in [
            ("Balanced", balanced),
            ("Conservative", conservative),
            ("Safe", safe),
        ]:
            if method in summary["comparison_to_baseline"]:
                agg = summary["comparison_to_baseline"][method]["wins"]
                cat = category_wins(summary, method)
                parts.append(f"{name}: {agg}/7 aggregate, {cat} category")
        if parts:
            rows.append(
                [
                    "The method transfers to an independently different inversion backbone.",
                    "ReNoise xcat10 reports "
                    + "; ".join(parts)
                    + "; file-hash check versus DDIM found 0/20 matching outputs.",
                    "Supported on cross-category ReNoise",
                ]
            )
    return rows


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def write_markdown(path, rows):
    header, *body = rows
    lines = [
        "| " + " | ".join(str(item) for item in header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def latex_escape(value):
    value = str(value)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def write_latex(path, rows, caption, label):
    header, *body = rows
    align = "l" * len(header)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(latex_escape(item) for item in header) + r" \\",
        r"\midrule",
    ]
    for row in body:
        lines.append(" & ".join(latex_escape(item) for item in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def column_name(index):
    name = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def sheet_xml(rows):
    sheet_rows = []
    for r_idx, row in enumerate(rows, start=1):
        cells = []
        for c_idx, value in enumerate(row):
            ref = f"{column_name(c_idx)}{r_idx}"
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                cells.append(f'<c r="{ref}"><v>{value}</v></c>')
            else:
                cells.append(
                    f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'
                )
        sheet_rows.append(f'<row r="{r_idx}">' + "".join(cells) + "</row>")
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        "<sheetData>"
        + "".join(sheet_rows)
        + "</sheetData></worksheet>"
    )


def safe_sheet_name(name):
    name = re.sub(r"[\[\]:*?/\\]", "_", name)
    return name[:31]


def write_xlsx(path, sheets):
    sheet_entries = []
    rel_entries = []
    content_overrides = []
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for index, (name, rows) in enumerate(sheets, start=1):
            sheet_path = f"xl/worksheets/sheet{index}.xml"
            zf.writestr(sheet_path, sheet_xml(rows))
            sheet_entries.append(
                f'<sheet name="{escape(safe_sheet_name(name))}" sheetId="{index}" r:id="rId{index}"/>'
            )
            rel_entries.append(
                f'<Relationship Id="rId{index}" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
                f'Target="worksheets/sheet{index}.xml"/>'
            )
            content_overrides.append(
                f'<Override PartName="/{sheet_path}" '
                'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            )
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            + "".join(content_overrides)
            + "</Types>",
        )
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        zf.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            "<sheets>"
            + "".join(sheet_entries)
            + "</sheets></workbook>",
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(rel_entries)
            + "</Relationships>",
        )


def write_table_bundle(prefix, rows, caption, label):
    write_csv(OUTPUT_DIR / f"{prefix}.csv", rows)
    write_markdown(OUTPUT_DIR / f"{prefix}.md", rows)
    write_latex(OUTPUT_DIR / f"{prefix}.tex", rows, caption, label)


def main():
    global SOURCE_DIR, OUTPUT_DIR
    args = parse_args()
    SOURCE_DIR = Path(args.source_dir)
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main_rows = main_table_rows()
    category_selection = ddim_category_selection_rows()
    npi_duplicate = npi_duplicate_rows()
    gnri = gnri_rows()
    renoise = renoise_rows() if has_source("renoise_xcat10_summary.json") else None
    ablation = ablation_rows()
    efficiency = efficiency_rows()
    reconstruction = reconstruction_delta_rows()
    claims = claim_evidence_rows()

    write_table_bundle(
        "tpami_table_main_full_piebench",
        main_rows,
        "Full PIE-Bench editing results on SDXL-DDIM.",
        "tab:tpami-main-full-piebench",
    )
    write_table_bundle(
        "tpami_table_npi_duplicate_check",
        npi_duplicate,
        "NPI sanity check. This run is not used as independent backbone evidence because file-level hashes match DDIM outputs under the current guidance setting.",
        "tab:tpami-npi-duplicate-check",
    )
    write_table_bundle(
        "tpami_table_ddim_category_selection",
        category_selection,
        "Category-level wins and selected schedule counts on full PIE-Bench with SDXL-DDIM. Schedule-count columns are ordered as TRDI/F50/F75/Late.",
        "tab:tpami-ddim-category-selection",
    )
    write_table_bundle(
        "tpami_table_gnri_xcat10",
        gnri,
        "GNRI cross-category stress test with 10 PIE-Bench samples per edit category.",
        "tab:tpami-gnri-xcat10",
    )
    if renoise:
        write_table_bundle(
            "tpami_table_renoise_xcat10",
            renoise,
            "ReNoise cross-category stress test with 10 PIE-Bench samples per edit category.",
            "tab:tpami-renoise-xcat10",
        )
    write_table_bundle(
        "tpami_table_selector_ablation",
        ablation,
        "Selector ablation on full PIE-Bench with SDXL-DDIM.",
        "tab:tpami-selector-ablation",
    )
    write_table_bundle(
        "tpami_table_efficiency",
        efficiency,
        "Observed wall-clock time and GPU-hours from audited run summaries. CGA-TRDI candidate wall time assumes four candidates run in parallel on four GPUs.",
        "tab:tpami-efficiency",
    )
    if len(reconstruction) > 1:
        write_table_bundle(
            "tpami_table_reconstruction_delta_audit",
            reconstruction,
            "COCO1000 reconstruction deltas under fresh-run protocols. These rows should be used as within-run evidence only, not as absolute reproduction of the original paper values.",
            "tab:tpami-reconstruction-delta-audit",
        )
    write_csv(OUTPUT_DIR / "tpami_claim_evidence.csv", claims)
    write_markdown(OUTPUT_DIR / "tpami_claim_evidence.md", claims)
    sheets = [
        ("Main Full PIE-Bench", main_rows),
        ("DDIM Category Selection", category_selection),
        ("NPI duplicate check", npi_duplicate),
        ("GNRI xcat10", gnri),
    ]
    if renoise:
        sheets.append(("ReNoise xcat10", renoise))
    sheets.extend(
        [
            ("Selector Ablation", ablation),
            ("Efficiency", efficiency),
            ("Recon Delta Audit", reconstruction),
            ("Claim Evidence", claims),
        ]
    )
    write_xlsx(OUTPUT_DIR / "tpami_extension_tables.xlsx", sheets)
    print(f"wrote tables to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
