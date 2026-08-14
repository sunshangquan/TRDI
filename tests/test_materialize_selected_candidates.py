import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "materialize_selected_candidates.py"


class MaterializeSelectedCandidatesTest(unittest.TestCase):
    def test_copies_the_selected_candidate_without_changing_bytes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            roots = {"trdi": root / "trdi", "late": root / "late"}
            relative_paths = [Path("group/first.jpg"), Path("group/second.jpg")]
            for candidate_root in roots.values():
                (candidate_root / "group").mkdir(parents=True)
            (roots["trdi"] / relative_paths[0]).write_bytes(b"trdi-first")
            (roots["late"] / relative_paths[0]).write_bytes(b"late-first")
            (roots["trdi"] / relative_paths[1]).write_bytes(b"trdi-second")
            (roots["late"] / relative_paths[1]).write_bytes(b"late-second")

            records_path = root / "selection_records.json"
            records_path.write_text(
                json.dumps(
                    [
                        {
                            "file": str(relative_paths[0]),
                            "selected_schedule_mode": "trdi",
                        },
                        {
                            "file": str(relative_paths[1]),
                            "selected_schedule_mode": "late",
                        },
                    ]
                ),
                encoding="utf-8",
            )
            output_root = root / "output"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--selection-records",
                    str(records_path),
                    "--candidate-image-roots",
                    f"trdi={roots['trdi']},late={roots['late']}",
                    "--output-root",
                    str(output_root),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertEqual(
                (output_root / relative_paths[0]).read_bytes(), b"trdi-first"
            )
            self.assertEqual(
                (output_root / relative_paths[1]).read_bytes(), b"late-second"
            )
            summary = json.loads(
                (output_root / "materialization_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["records"], 2)
            self.assertEqual(summary["selected_schedule_counts"], {"late": 1, "trdi": 1})
            self.assertEqual(summary["copy_mode"], "byte_preserving")

    def test_rejects_parent_directory_in_record_path(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            candidate_root = root / "trdi"
            candidate_root.mkdir()
            records_path = root / "selection_records.json"
            records_path.write_text(
                json.dumps(
                    [
                        {
                            "file": "../outside.jpg",
                            "selected_schedule_mode": "trdi",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--selection-records",
                    str(records_path),
                    "--candidate-image-roots",
                    f"trdi={candidate_root}",
                    "--output-root",
                    str(root / "output"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unsafe relative path", result.stderr)


if __name__ == "__main__":
    unittest.main()
