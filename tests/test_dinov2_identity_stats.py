import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_dinov2_identity.py"
SPEC = importlib.util.spec_from_file_location("evaluate_dinov2_identity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_summary_preserves_direction_and_exact_ties():
    result = MODULE.summarize_improvements(
        np.array([0.2, 0.1, 0.0, -0.05]), bootstrap_resamples=200, seed=9
    )
    assert np.isclose(result["mean_improvement"], 0.0625)
    assert result["wins"] == 2
    assert result["ties"] == 1
    assert result["losses"] == 1
    assert len(result["bootstrap_ci95"]) == 2
