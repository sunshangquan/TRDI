import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "paired_significance.py"
SPEC = importlib.util.spec_from_file_location("paired_significance", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PairedSignificanceTest(unittest.TestCase):
    def test_holm_adjustment_is_monotone_in_sorted_pvalue_order(self):
        pvalues = np.array([0.04, 0.001, 0.02])
        adjusted = MODULE.holm_adjust(pvalues)
        order = np.argsort(pvalues)
        self.assertTrue(np.all(np.diff(adjusted[order]) >= 0))
        np.testing.assert_allclose(adjusted, [0.04, 0.003, 0.04])

    def test_analysis_uses_improvement_direction_and_exact_ties(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline = root / "baseline.csv"
            method = root / "method.csv"
            fields = ["file_id", *[metric for metric, _ in MODULE.METRICS]]
            baseline_rows = []
            method_rows = []
            for index in range(4):
                baseline_row = {"file_id": str(index)}
                method_row = {"file_id": str(index)}
                for metric, higher_is_better in MODULE.METRICS:
                    baseline_row[metric] = 1.0
                    if index == 0:
                        method_row[metric] = 1.0
                    else:
                        method_row[metric] = 2.0 if higher_is_better else 0.0
                baseline_rows.append(baseline_row)
                method_rows.append(method_row)
            for path, rows in ((baseline, baseline_rows), (method, method_rows)):
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)

            result = MODULE.analyze(baseline, method, bootstrap_resamples=100, seed=7)
            self.assertEqual(result["paired_file_ids"], 4)
            for metric in result["metrics"]:
                self.assertEqual(metric["wins"], 3)
                self.assertEqual(metric["ties"], 1)
                self.assertEqual(metric["losses"], 0)
                self.assertEqual(metric["mean_improvement"], 0.75)


if __name__ == "__main__":
    unittest.main()
