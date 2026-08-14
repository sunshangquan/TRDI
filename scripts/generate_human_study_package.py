#!/usr/bin/env python3

import argparse
import csv
import hashlib
import html
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


QUESTIONS = (
    ("edit_choice", "Which output follows the requested edit better?"),
    ("preserve_choice", "Which output better preserves unrelated content?"),
    ("overall_choice", "Which output is better overall?"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build blinded, stratified TRDI/CGA-TRDI human-study forms."
    )
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--trdi-root", type=Path, required=True)
    parser.add_argument("--cga-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--participants", type=int, default=3)
    parser.add_argument("--per-category", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--attention-checks", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260814)
    return parser.parse_args()


def category_name(record):
    image_path = Path(record["image_path"])
    return image_path.parts[0] if len(image_path.parts) > 1 else record["editing_type_id"]


def select_cases(mapping, per_category, seed):
    grouped = defaultdict(list)
    for file_id, record in mapping.items():
        grouped[category_name(record)].append((file_id, record))
    rng = random.Random(seed)
    selected = []
    for category in sorted(grouped):
        cases = sorted(grouped[category])
        if len(cases) < per_category:
            raise ValueError(
                f"category {category!r} has {len(cases)} cases, needs {per_category}"
            )
        selected.extend(rng.sample(cases, per_category))
    return selected


def require_image(root, relative_path):
    path = root / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def anonymized_name(role, relative_path, source_path):
    token = hashlib.sha256(
        f"{role}:{relative_path}".encode("utf-8") + source_path.read_bytes()
    ).hexdigest()[:20]
    return f"{token}{source_path.suffix.lower()}"


def copy_asset(source, asset_root, role, relative_path):
    name = anonymized_name(role, relative_path, source)
    destination = asset_root / name
    if not destination.exists():
        shutil.copy2(source, destination)
    return f"assets/{name}"


def render_html(participant_id, rows):
    question_html = []
    for row in rows:
        controls = []
        for field, question in QUESTIONS:
            controls.append(
                f'<fieldset><legend>{html.escape(question)}</legend>'
                + "".join(
                    f'<label><input required type="radio" name="{field}_{row["trial_id"]}" '
                    f'value="{choice}"> {choice}</label>'
                    for choice in ("Left", "Right", "Tie")
                )
                + "</fieldset>"
            )
        question_html.append(
            f'''<article class="trial" data-trial="{row["trial_id"]}"
                 data-case="{row["case_id"]}" data-category="{html.escape(row["category"])}"
                 data-repeat="{int(row["is_repeat"])}" data-attention="{int(row["is_attention_check"])}">
              <header><span>Case {row["display_index"]} / {len(rows)}</span></header>
              <p class="instruction"><strong>Edit request:</strong> {html.escape(row["instruction"])}</p>
              <p><strong>Target prompt:</strong> {html.escape(row["target_prompt"])}</p>
              <div class="images">
                <figure><img src="../{row["source_asset"]}" alt="Source"><figcaption>Source</figcaption></figure>
                <figure><img src="../{row["left_asset"]}" alt="Left output"><figcaption>Left</figcaption></figure>
                <figure><img src="../{row["right_asset"]}" alt="Right output"><figcaption>Right</figcaption></figure>
              </div>
              <div class="questions">{"".join(controls)}</div>
            </article>'''
        )
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Image Editing Preference Study</title>
<style>
body{{font:15px Arial,sans-serif;margin:0;background:#f4f5f7;color:#181a1f}} main{{max-width:1180px;margin:auto;padding:24px}}
.trial{{background:white;border:1px solid #d8dbe1;border-radius:6px;padding:18px;margin:0 0 22px}}
header{{color:#555}} .instruction{{font-size:17px}} .images{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}
figure{{margin:0}} img{{display:block;width:100%;aspect-ratio:1/1;object-fit:contain;background:#eee}} figcaption{{text-align:center;font-weight:bold;margin-top:6px}}
.questions{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-top:16px}} fieldset{{border:1px solid #d8dbe1}}
label{{display:inline-block;margin:8px 14px 4px 0}} button{{padding:10px 16px;font-weight:bold}}
@media(max-width:760px){{.images,.questions{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Image Editing Preference Study</h1>
<p>Participant: <strong>{html.escape(participant_id)}</strong>. Judge the anonymous outputs from the images only.</p>
<form id="study">{"".join(question_html)}<button type="submit">Download responses CSV</button></form>
<script>
const participant={json.dumps(participant_id)};
const startedAt=Date.now();
document.getElementById('study').addEventListener('submit', e=>{{e.preventDefault();
 const header=['participant_id','trial_id','case_id','category','is_repeat','is_attention_check','edit_choice','preserve_choice','overall_choice','session_seconds','submitted_at_utc'];
 const sessionSeconds=Math.round((Date.now()-startedAt)/1000); const submittedAt=new Date().toISOString();
 const lines=[header]; document.querySelectorAll('.trial').forEach(t=>{{
  const id=t.dataset.trial; const val=f=>document.querySelector(`input[name="${{f}}_${{id}}"]:checked`).value;
  lines.push([participant,id,t.dataset.case,t.dataset.category,t.dataset.repeat,t.dataset.attention,val('edit_choice'),val('preserve_choice'),val('overall_choice'),sessionSeconds,submittedAt]);
 }}); const csv=lines.map(r=>r.map(v=>'"'+String(v).replaceAll('"','""')+'"').join(',')).join('\\n');
 const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([csv],{{type:'text/csv'}}));
 a.download=participant+'_responses.csv'; a.click(); URL.revokeObjectURL(a.href);
}});
</script></main></body></html>'''


def main():
    args = parse_args()
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    selected = select_cases(mapping, args.per_category, args.seed)
    output_root = args.output_root.resolve()
    public_root = output_root / "public"
    asset_root = public_root / "assets"
    form_root = public_root / "forms"
    private_root = output_root / "private"
    asset_root.mkdir(parents=True, exist_ok=True)
    form_root.mkdir(parents=True, exist_ok=True)
    private_root.mkdir(parents=True, exist_ok=True)

    cases = []
    for file_id, record in selected:
        relative = Path(record["image_path"])
        source = require_image(args.source_root, relative)
        trdi = require_image(args.trdi_root, relative)
        cga = require_image(args.cga_root, relative)
        cases.append(
            {
                "case_id": file_id,
                "category": category_name(record),
                "instruction": record["editing_instruction"],
                "target_prompt": record["editing_prompt"].replace("[", "").replace("]", ""),
                "source_asset": copy_asset(source, asset_root, "source", relative),
                "trdi_asset": copy_asset(trdi, asset_root, "trdi", relative),
                "cga_asset": copy_asset(cga, asset_root, "cga", relative),
            }
        )

    manifest_rows = []
    for participant_index in range(args.participants):
        participant_id = f"rater_{participant_index + 1:02d}"
        rng = random.Random(args.seed + 10_000 + participant_index)
        participant_cases = [
            dict(case, is_repeat=False, is_attention_check=False) for case in cases
        ]
        repeats = rng.sample(cases, min(args.repeats, len(cases)))
        participant_cases.extend(
            dict(case, is_repeat=True, is_attention_check=False) for case in repeats
        )
        attention_cases = rng.sample(cases, min(args.attention_checks, len(cases)))
        participant_cases.extend(
            dict(case, is_repeat=False, is_attention_check=True)
            for case in attention_cases
        )
        rng.shuffle(participant_cases)
        rendered_rows = []
        for trial_index, case in enumerate(participant_cases, start=1):
            if case["is_attention_check"]:
                left_method = right_method = "Identical"
                left_asset = right_asset = case["cga_asset"]
            else:
                trdi_left = bool(rng.getrandbits(1))
                left_method, right_method = (
                    ("TRDI", "CGA-TRDI") if trdi_left else ("CGA-TRDI", "TRDI")
                )
                left_asset = case["trdi_asset"] if trdi_left else case["cga_asset"]
                right_asset = case["cga_asset"] if trdi_left else case["trdi_asset"]
            trial_id = f"{participant_id}_t{trial_index:03d}"
            rendered_rows.append(
                dict(
                    case,
                    trial_id=trial_id,
                    display_index=trial_index,
                    left_asset=left_asset,
                    right_asset=right_asset,
                )
            )
            manifest_rows.append(
                {
                    "participant_id": participant_id,
                    "trial_id": trial_id,
                    "case_id": case["case_id"],
                    "category": case["category"],
                    "is_repeat": int(case["is_repeat"]),
                    "is_attention_check": int(case["is_attention_check"]),
                    "left_method": left_method,
                    "right_method": right_method,
                    "target_prompt": case["target_prompt"],
                    "source_asset": case["source_asset"],
                    "left_asset": left_asset,
                    "right_asset": right_asset,
                }
            )
        (form_root / f"{participant_id}.html").write_text(
            render_html(participant_id, rendered_rows), encoding="utf-8"
        )

    manifest_path = private_root / "randomization_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    summary = {
        "seed": args.seed,
        "categories": len({case["category"] for case in cases}),
        "unique_cases": len(cases),
        "participants": args.participants,
        "repeats_per_participant": args.repeats,
        "attention_checks_per_participant": args.attention_checks,
        "trials_per_participant": len(cases) + args.repeats + args.attention_checks,
        "public_forms": str(form_root),
        "private_manifest": str(manifest_path),
    }
    (output_root / "package_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
