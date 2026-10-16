from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss


def multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray, n_classes: int) -> float:
    one_hot = np.eye(n_classes, dtype=np.float32)[y_true]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def adaptive_ece(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    """Equal-mass ECE for multiclass confidence.

    It is intentionally reported with NLL and Brier score. ECE alone is not used as a
    probability-quality claim because its value depends on binning and cohort size.
    """
    conf = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == y_true).astype(float)
    boundaries = np.unique(np.quantile(conf, np.linspace(0, 1, bins + 1)))
    if len(boundaries) <= 2:
        return float(abs(conf.mean() - correct.mean()))
    ece = 0.0
    for lo, hi in pairwise(boundaries):
        mask = (conf >= lo) & (conf <= hi if hi == boundaries[-1] else conf < hi)
        if mask.any():
            ece += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())
    return float(ece)


def risk_coverage_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    confidence = probabilities.max(axis=1)
    correctness = (probabilities.argmax(axis=1) == y_true).astype(float)
    order = np.argsort(-confidence, kind="stable")
    risks = []
    for k in range(1, len(order) + 1):
        risks.append(1.0 - correctness[order[:k]].mean())
    coverage = np.arange(1, len(order) + 1) / len(order)
    return float(np.trapezoid(risks, coverage))


def _episodes(mask: np.ndarray) -> list[tuple[int, int]]:
    episodes: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(mask.astype(bool)):
        if value and start is None:
            start = i
        elif not value and start is not None:
            episodes.append((start, i - 1))
            start = None
    if start is not None:
        episodes.append((start, len(mask) - 1))
    return episodes


def _ordered_group_indices(
    group_ids: np.ndarray | None,
    time_order: np.ndarray | None,
    n_samples: int,
) -> list[np.ndarray]:
    """Return stable time-ordered indices within each participant/session trace.

    Episode and boundary metrics are invalid when the input table happens to be shuffled.
    Group boundaries are explicit so no temporal post-processing can bridge subjects or
    separate sessions.
    """
    if group_ids is None:
        group_ids = np.zeros(n_samples, dtype=int)
    groups = pd.Series(group_ids).astype(str).to_numpy()
    if time_order is None:
        time_order = np.arange(n_samples, dtype=float)
    values = np.asarray(time_order, dtype=float)
    if values.shape != (n_samples,):
        raise ValueError("time_order must have shape [N]")
    if np.any(~np.isfinite(values)):
        raise ValueError("time_order must be finite")
    ordered: list[np.ndarray] = []
    for group in pd.Series(groups).drop_duplicates().tolist():
        idx = np.flatnonzero(groups == group)
        local = idx[np.argsort(values[idx], kind="stable")]
        ordered.append(local)
    return ordered


def episode_metrics(
    y_true: np.ndarray,
    stress_score: np.ndarray,
    *,
    threshold: float,
    hop_seconds: float,
    elevated_class: int = 1,
    group_ids: np.ndarray | None = None,
    time_order: np.ndarray | None = None,
) -> dict[str, float]:
    """Episode metrics that reset and sort at participant/session boundaries."""
    if len(y_true) != len(stress_score):
        raise ValueError("y_true and stress_score must have matching length")
    truth = y_true == elevated_class
    pred = stress_score >= threshold
    touched = 0
    n_true = 0
    n_pred = 0
    onset_errors: list[float] = []
    offset_errors: list[float] = []
    for idx in _ordered_group_indices(group_ids, time_order, len(y_true)):
        local_truth = truth[idx]
        local_pred = pred[idx]
        true_eps = _episodes(local_truth)
        pred_eps = _episodes(local_pred)
        n_true += len(true_eps)
        n_pred += len(pred_eps)
        for ts, te in true_eps:
            overlaps = [(ps, pe) for ps, pe in pred_eps if not (pe < ts or ps > te)]
            if overlaps:
                touched += 1
                # Select the overlap with greatest temporal intersection, not arbitrary first match.
                ps, pe = max(overlaps, key=lambda pair: min(pair[1], te) - max(pair[0], ts) + 1)
                onset_errors.append(abs(ps - ts) * hop_seconds)
                offset_errors.append(abs(pe - te) * hop_seconds)
    recall = touched / n_true if n_true else 0.0
    nonstress = ~truth
    false_alert_minutes = float((pred & nonstress).sum() * hop_seconds / 60.0)
    nonstress_hours = max(float(nonstress.sum() * hop_seconds / 3600.0), 1e-6)
    false_alert_per_hour = false_alert_minutes / nonstress_hours
    pred_minutes = float(pred.sum() * hop_seconds / 60.0)
    return {
        "threshold": float(threshold),
        "stress_episode_recall": float(recall),
        "n_true_episodes": float(n_true),
        "n_pred_episodes": float(n_pred),
        "false_alert_minutes_per_nonstress_hour": float(false_alert_per_hour),
        "false_alert_minutes": false_alert_minutes,
        "predicted_alert_minutes": pred_minutes,
        "median_onset_error_s": float(np.median(onset_errors)) if onset_errors else float("nan"),
        "median_offset_error_s": float(np.median(offset_errors)) if offset_errors else float("nan"),
    }


def threshold_frontier(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: list[float],
    hop_seconds: float,
    elevated_class: int = 1,
    group_ids: np.ndarray | None = None,
    time_order: np.ndarray | None = None,
) -> pd.DataFrame:
    rows = [
        episode_metrics(
            y_true,
            probabilities[:, elevated_class],
            threshold=t,
            hop_seconds=hop_seconds,
            elevated_class=elevated_class,
            group_ids=group_ids,
            time_order=time_order,
        )
        for t in thresholds
    ]
    return pd.DataFrame(rows)


def choose_operating_threshold(
    frontier: pd.DataFrame, target_minutes_per_hour: float
) -> dict[str, float]:
    """Choose an operating threshold on development data only.

    The returned threshold must be frozen before any untouched final or external dataset is
    evaluated. The final-set frontier may still be plotted as a diagnostic, but may not select
    the reported operating point.
    """
    if frontier.empty:
        raise ValueError("threshold frontier cannot be empty")
    eligible = frontier[
        frontier["false_alert_minutes_per_nonstress_hour"] <= target_minutes_per_hour
    ]
    if eligible.empty:
        selected = frontier.iloc[frontier["false_alert_minutes_per_nonstress_hour"].argmin()]
        status = "no_threshold_meets_dev_burden"
    else:
        selected = eligible.sort_values(
            ["stress_episode_recall", "false_alert_minutes_per_nonstress_hour", "threshold"],
            ascending=[False, True, True],
        ).iloc[0]
        status = "meets_dev_burden"
    result = {k: float(v) for k, v in selected.to_dict().items()}
    result["selection_status"] = status
    result["target_false_alert_minutes_per_hour"] = float(target_minutes_per_hour)
    return result


def _base_predictive_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, *, ece_bins: int
) -> dict[str, float]:
    pred = probabilities.argmax(axis=1)
    n_classes = probabilities.shape[1]
    return {
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "stress_f1": float(f1_score(y_true == 1, pred == 1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "nll": float(log_loss(y_true, probabilities, labels=np.arange(n_classes))),
        "brier": multiclass_brier(y_true, probabilities, n_classes),
        "adaptive_ece": adaptive_ece(y_true, probabilities, bins=ece_bins),
        "aurc": risk_coverage_auc(y_true, probabilities),
    }


def evaluate_predictions(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    hop_seconds: float,
    thresholds: list[float],
    target_false_alert_minutes_per_hour: float,
    elevated_class: int = 1,
    ece_bins: int = 10,
    group_ids: np.ndarray | None = None,
    time_order: np.ndarray | None = None,
    operating_threshold: float | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate fixed predictions.

    `operating_threshold=None` is allowed for development threshold selection only. A final or
    external caller must pass a threshold selected on a disjoint development cohort.
    """
    if probabilities.ndim != 2 or len(y_true) != len(probabilities):
        raise ValueError("invalid y_true/probabilities dimensions")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("probabilities must sum to one")
    metrics = _base_predictive_metrics(y_true, probabilities, ece_bins=ece_bins)
    frontier = threshold_frontier(
        y_true,
        probabilities,
        thresholds,
        hop_seconds,
        elevated_class,
        group_ids,
        time_order,
    )
    if operating_threshold is not None:
        fixed = episode_metrics(
            y_true,
            probabilities[:, elevated_class],
            threshold=float(operating_threshold),
            hop_seconds=hop_seconds,
            elevated_class=elevated_class,
            group_ids=group_ids,
            time_order=time_order,
        )
        metrics.update({f"operating_{k}": v for k, v in fixed.items()})
    else:
        selection = choose_operating_threshold(frontier, target_false_alert_minutes_per_hour)
        metrics.update(
            {
                f"development_selection_{k}": v
                for k, v in selection.items()
                if k != "selection_status"
            }
        )
    return metrics, frontier


def selective_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    defer: np.ndarray,
    *,
    elevated_class: int = 1,
) -> dict[str, float]:
    """Metrics for the abstention-aware production surface.

    These values intentionally separate all-row predictive quality from the quality of rows the
    system actually supports. A low-risk selective metric with near-zero coverage is not enough,
    so coverage is emitted alongside it.
    """
    if defer.shape != y_true.shape:
        raise ValueError("defer must have shape [N]")
    accepted = ~defer.astype(bool)
    coverage = float(accepted.mean())
    if not accepted.any():
        return {
            "selective_coverage": coverage,
            "defer_rate": 1.0,
            "selective_macro_f1": float("nan"),
            "selective_stress_f1": float("nan"),
            "selective_error_rate": float("nan"),
            "unsupported_elevated_rate": 0.0,
        }
    pred = probabilities.argmax(axis=1)
    elevated = pred == elevated_class
    return {
        "selective_coverage": coverage,
        "defer_rate": float(defer.astype(bool).mean()),
        "selective_macro_f1": float(
            f1_score(y_true[accepted], pred[accepted], average="macro", zero_division=0)
        ),
        "selective_stress_f1": float(
            f1_score(
                y_true[accepted] == elevated_class,
                pred[accepted] == elevated_class,
                zero_division=0,
            )
        ),
        "selective_error_rate": float((pred[accepted] != y_true[accepted]).mean()),
        "unsupported_elevated_rate": float((elevated & defer.astype(bool)).mean()),
    }


def slice_metrics(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    **kwargs: Any,
) -> pd.DataFrame:
    rows = []
    for subject, idx in metadata.groupby("subject_id", sort=True).groups.items():
        indices = np.asarray(list(idx), dtype=int)
        local_kwargs = dict(kwargs)
        local_kwargs["group_ids"] = np.zeros(len(indices), dtype=int)
        local_kwargs["time_order"] = metadata.iloc[indices]["window_end_s"].to_numpy(dtype=float)
        result, _ = evaluate_predictions(y_true[indices], probabilities[indices], **local_kwargs)
        result["slice_type"] = "subject"
        result["slice_value"] = str(subject)
        result["n_samples"] = len(indices)
        rows.append(result)
    return pd.DataFrame(rows)


def paired_subject_bootstrap(
    subject_values: pd.DataFrame,
    baseline_col: str,
    challenger_col: str,
    *,
    resamples: int = 1000,
    seed: int = 17,
) -> dict[str, float]:
    values = subject_values[[baseline_col, challenger_col]].dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return {
            "uplift": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_subjects": 0.0,
        }
    diff = values[:, 1] - values[:, 0]
    rng = np.random.default_rng(seed)
    boot = np.array(
        [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(resamples)]
    )
    return {
        "uplift": float(diff.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "n_subjects": float(len(values)),
    }
