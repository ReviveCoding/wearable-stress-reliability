from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from healthsense.datasets.contracts import CanonicalBatch
from healthsense.evaluation.metrics import evaluate_predictions
from healthsense.quality.gates import apply_modality_dropout, inject_motion_noise, inject_timing_lag


def run_robustness_suite(
    predict: Callable[[CanonicalBatch], Any],
    batch: CanonicalBatch,
    *,
    evaluation_kwargs: dict[str, Any],
) -> pd.DataFrame:
    scenarios: list[tuple[str, CanonicalBatch]] = [("clean", batch)]
    for modality in range(batch.modality_mask.shape[1]):
        scenarios.append((f"dropout_modality_{modality}", apply_modality_dropout(batch, modality)))
        scenarios.append((f"lag15s_modality_{modality}", inject_timing_lag(batch, modality, 15000)))
    scenarios.append(("motion_noise", inject_motion_noise(batch, seed=23)))
    rows = []
    for name, corrupted in scenarios:
        bundle = predict(corrupted)
        metrics, _ = evaluate_predictions(
            corrupted.labels, bundle.probabilities, **evaluation_kwargs
        )
        metrics["scenario"] = name
        rows.append(metrics)
    return pd.DataFrame(rows)
