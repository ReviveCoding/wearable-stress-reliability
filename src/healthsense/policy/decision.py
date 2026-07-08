from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def apply_selective_policy(
    probabilities: np.ndarray,
    quality: np.ndarray,
    *,
    confidence_threshold: float,
    quality_threshold: float,
    elevated_class: int = 1,
) -> pd.DataFrame:
    if probabilities.ndim != 2 or quality.ndim != 2 or len(probabilities) != len(quality):
        raise ValueError("probabilities and quality must have matching sample dimensions")
    pred = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    quality_mean = quality.mean(axis=1)
    defer = (confidence < confidence_threshold) | (quality_mean < quality_threshold)
    action = np.where(
        defer,
        "DEFER_LOW_CONFIDENCE_OR_LOW_SIGNAL_QUALITY",
        np.where(pred == elevated_class, "ELEVATED_STATE_SUPPORTED", "NO_ELEVATED_STATE_SUPPORTED"),
    )
    return pd.DataFrame(
        {
            "predicted_state": pred,
            "confidence": confidence,
            "quality_mean": quality_mean,
            "defer": defer.astype(int),
            "action": action,
        }
    )


def _candidate(comparison: pd.DataFrame, candidate_id: str) -> pd.Series | None:
    if "candidate_id" not in comparison.columns:
        return None
    rows = comparison.loc[comparison["candidate_id"] == candidate_id]
    return None if rows.empty else rows.iloc[0]


def release_decision(
    comparison: pd.DataFrame,
    robustness: pd.DataFrame,
    operational: pd.DataFrame,
    *,
    gates: dict[str, float],
    predictive_champion: str,
    production_champion: str,
    statistical_comparison: dict[str, Any] | None = None,
    slice_comparison: pd.DataFrame | None = None,
    selective: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Use held-out, preselected operating points and explicit reliability gates.

    C3 may be the predictive challenger while C5 is the calibrated routing surface. The decision
    therefore separates specialty uplift from production calibration and abstention behavior.
    """
    reasons: list[str] = []
    baseline = _candidate(comparison, "C2")
    predictive = _candidate(comparison, predictive_champion)
    production = _candidate(comparison, production_champion)
    if baseline is None or predictive is None or production is None:
        return {
            "disposition": "BLOCK_FOR_INTEGRITY_OR_SCOPE_VIOLATION",
            "reasons": ["C2, predictive champion, or production champion unavailable"],
        }

    uplift_pp = 100 * (
        predictive["operating_stress_episode_recall"] - baseline["operating_stress_episode_recall"]
    )
    macro_decline_pp = 100 * (baseline["macro_f1"] - predictive["macro_f1"])
    ece_regression = production["adaptive_ece"] - baseline["adaptive_ece"]
    if uplift_pp < gates["min_episode_recall_uplift_pp"]:
        reasons.append(f"predictive episode recall uplift {uplift_pp:.2f}pp below gate")
    if macro_decline_pp > gates["max_macro_f1_decline_pp"]:
        reasons.append(f"predictive macro-F1 decline {macro_decline_pp:.2f}pp exceeds gate")
    if ece_regression > gates["max_ece_regression"]:
        reasons.append(f"production ECE regression {ece_regression:.4f} exceeds gate")

    ci_low = float("nan")
    if statistical_comparison is None:
        reasons.append("missing participant-bootstrap comparison")
    else:
        ci_low = float(statistical_comparison.get("ci_low", float("nan")))
        n_bootstrap_subjects = float(statistical_comparison.get("n_subjects", 0.0))
        minimum_subjects = float(gates.get("min_bootstrap_subjects", 1.0))
        if n_bootstrap_subjects < minimum_subjects:
            reasons.append(
                f"only {n_bootstrap_subjects:.0f} final subjects; at least {minimum_subjects:.0f} are required for a release claim"
            )
        if not np.isfinite(ci_low) or ci_low <= 0.0:
            reasons.append(
                f"participant-bootstrap lower CI {ci_low:.4f} does not support positive uplift"
            )

    worst_subject_decline_pp = float("nan")
    if slice_comparison is None or slice_comparison.empty:
        reasons.append("missing worst-subject comparison")
    else:
        required = {"C2", predictive_champion}
        if not required.issubset(slice_comparison.columns):
            reasons.append("incomplete worst-subject comparison")
        else:
            deltas = slice_comparison[predictive_champion] - slice_comparison["C2"]
            worst_subject_decline_pp = float(max(0.0, -100.0 * deltas.min()))
            if worst_subject_decline_pp > gates["max_worst_subject_recall_decline_pp"]:
                reasons.append(
                    f"worst-subject recall decline {worst_subject_decline_pp:.2f}pp exceeds gate"
                )

    clean = (
        robustness[robustness["scenario"] == "clean"]
        if "scenario" in robustness.columns
        else pd.DataFrame()
    )
    fault = (
        robustness[robustness["scenario"] != "clean"]
        if "scenario" in robustness.columns
        else pd.DataFrame()
    )
    if not clean.empty and not fault.empty:
        clean_f1 = float(clean.iloc[0]["macro_f1"])
        worst_fault_f1 = float(fault["macro_f1"].min())
        relative_degradation = (clean_f1 - worst_fault_f1) / max(clean_f1, 1e-6)
        if relative_degradation > gates["max_relative_fault_degradation"]:
            reasons.append(
                f"production fault macro-F1 degradation {relative_degradation:.3f} exceeds gate"
            )
    else:
        relative_degradation = float("nan")
        reasons.append("missing production robustness metrics")

    op = _candidate(operational, production_champion)
    if op is None:
        reasons.append("missing production operational metrics")
    elif float(op["inference_p95_ms"]) > gates["max_inference_p95_ms"]:
        reasons.append("production inference p95 latency exceeds gate")

    selective_row = _candidate(
        selective if selective is not None else pd.DataFrame(), production_champion
    )
    selective_coverage = float("nan")
    if selective_row is None:
        reasons.append("missing selective policy metrics")
    else:
        selective_coverage = float(selective_row["selective_coverage"])
        max_defer = gates.get("max_defer_rate")
        if max_defer is not None and float(selective_row["defer_rate"]) > max_defer:
            reasons.append(
                f"production defer rate {float(selective_row['defer_rate']):.3f} exceeds gate"
            )

    disposition = (
        "ADVANCE_TO_CONTROLLED_RESEARCH_VALIDATION" if not reasons else "HOLD_FOR_FAILURE_ANALYSIS"
    )
    return {
        "disposition": disposition,
        "predictive_champion": predictive_champion,
        "production_champion": production_champion,
        "gates": gates,
        "values": {
            "predictive_episode_recall_uplift_pp_vs_C2": float(uplift_pp),
            "predictive_macro_f1_decline_pp_vs_C2": float(macro_decline_pp),
            "production_ece_regression_vs_C2": float(ece_regression),
            "participant_bootstrap_ci_low": ci_low,
            "worst_subject_recall_decline_pp": worst_subject_decline_pp,
            "production_relative_fault_degradation": float(relative_degradation),
            "production_selective_coverage": selective_coverage,
        },
        "reasons": reasons,
        "claim_boundary": "Synthetic-fixture evidence cannot support public-data performance or clinical claims.",
    }
