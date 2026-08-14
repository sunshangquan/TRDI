import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "generate_human_study_package.py"
ANALYZER = ROOT / "scripts" / "analyze_human_study.py"


def make_image(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


def test_package_is_blinded_and_analysis_unblinds(tmp_path):
    mapping = {}
    source_root = tmp_path / "source"
    trdi_root = tmp_path / "trdi"
    cga_root = tmp_path / "cga"
    for index in range(4):
        category = f"{index // 2}_category"
        relative = f"{category}/{index:03d}.png"
        mapping[str(index)] = {
            "image_path": relative,
            "editing_type_id": str(index // 2),
            "editing_instruction": f"edit {index}",
            "editing_prompt": f"a [target {index}]",
        }
        make_image(source_root / relative, (index, 0, 0))
        make_image(trdi_root / relative, (0, index, 0))
        make_image(cga_root / relative, (0, 0, index))
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    package_root = tmp_path / "study"
    subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--mapping",
            str(mapping_path),
            "--source-root",
            str(source_root),
            "--trdi-root",
            str(trdi_root),
            "--cga-root",
            str(cga_root),
            "--output-root",
            str(package_root),
            "--participants",
            "1",
            "--per-category",
            "2",
            "--repeats",
            "1",
            "--attention-checks",
            "1",
        ],
        check=True,
    )
    form = (package_root / "public/forms/rater_01.html").read_text(encoding="utf-8")
    assert "CGA-TRDI" not in form
    assert ">TRDI<" not in form
    manifest_path = package_root / "private/randomization_manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    assert len(manifest) == 6

    response_path = tmp_path / "responses.csv"
    fieldnames = [
        "participant_id",
        "trial_id",
        "case_id",
        "category",
        "is_repeat",
        "is_attention_check",
        "edit_choice",
        "preserve_choice",
        "overall_choice",
    ]
    with response_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest:
            cga_side = (
                "Tie"
                if row["is_attention_check"] == "1"
                else "Left" if row["left_method"] == "CGA-TRDI" else "Right"
            )
            writer.writerow(
                {
                    **{key: row[key] for key in fieldnames[:6]},
                    "edit_choice": cga_side,
                    "preserve_choice": cga_side,
                    "overall_choice": cga_side,
                }
            )
    output_path = tmp_path / "analysis.json"
    subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            "--manifest",
            str(manifest_path),
            "--responses",
            str(response_path),
            "--output",
            str(output_path),
            "--bootstrap-resamples",
            "100",
        ],
        check=True,
    )
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["outcomes"]["overall_choice"]["cga_preference_with_half_ties"] == 1.0
    assert result["outcomes"]["overall_choice"]["unique_cases"] == 4
