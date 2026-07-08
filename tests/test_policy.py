import numpy as np
import pandas as pd

from healthsense.policy.decision import apply_selective_policy, release_decision


def test_selective_policy_defer():
    p = np.array([[0.4, 0.5, 0.1], [0.9, 0.05, 0.05]])
    q = np.array([[0.2, 0.2], [0.9, 0.9]])
    out = apply_selective_policy(p, q, confidence_threshold=0.6, quality_threshold=0.3)
    assert out.loc[0, "action"] == "DEFER_LOW_CONFIDENCE_OR_LOW_SIGNAL_QUALITY"
    assert out.loc[1, "action"] == "NO_ELEVATED_STATE_SUPPORTED"


def test_release_blocks_missing_champion():
    result = release_decision(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        gates={},
        predictive_champion="C3",
        production_champion="C5",
    )
    assert result["disposition"].startswith("BLOCK")
