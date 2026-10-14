from __future__ import annotations

from copy import deepcopy
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from healthsense.utils.io import stable_json_hash


_REQUIRED_TOP_LEVEL = {"run", "data", "models", "evaluation", "policy"}


def _finite_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer, not boolean")
    parsed = _finite_float(value, label)
    if parsed <= 0 or not parsed.is_integer():
        raise ValueError(f"{label} must be a positive integer")
    return int(parsed)


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer, not boolean")
    parsed = _finite_float(value, label)
    if parsed < 0 or not parsed.is_integer():
        raise ValueError(f"{label} must be a non-negative integer")
    return int(parsed)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must decode to a mapping: {path}")
    return payload


def _default_config() -> tuple[dict[str, Any], str]:
    resource = resources.files("healthsense.resources").joinpath("smoke.yaml")
    with resource.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):  # pragma: no cover - packaged static resource
        raise ValueError("packaged default configuration must decode to a mapping")
    return payload, "package://healthsense.resources/smoke.yaml"


def validate_config(config: dict[str, Any]) -> None:
    missing = sorted(_REQUIRED_TOP_LEVEL - set(config))
    if missing:
        raise ValueError(f"configuration missing required sections: {missing}")
    for section in _REQUIRED_TOP_LEVEL:
        if not isinstance(config[section], dict):
            raise ValueError(f"configuration section '{section}' must be a mapping")

    data = config["data"]
    models = config["models"]
    evaluation = config["evaluation"]
    policy = config["policy"]
    run = config["run"]

    seed = run.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("run.seed must be an integer, not a boolean")
    if not isinstance(run.get("run_id"), str) or not run["run_id"].strip():
        raise ValueError("run.run_id must be a non-empty string")
    if not isinstance(run.get("output_root"), str) or not run["output_root"].strip():
        raise ValueError("run.output_root must be a non-empty string")
    if not isinstance(run.get("deterministic"), bool):
        raise ValueError("run.deterministic must be a boolean")
    data_ints = {
        key: _positive_int(data.get(key, 0), f"data.{key}")
        for key in (
            "window_seconds",
            "hop_seconds",
            "n_subjects",
            "windows_per_subject",
            "sequence_length",
            "n_modalities",
            "n_channels",
        )
    }
    if data_ints["n_channels"] % data_ints["n_modalities"] != 0:
        raise ValueError("data.n_channels must be divisible by data.n_modalities")
    final_holdout = _positive_int(
        data.get("final_holdout_subjects", 0), "data.final_holdout_subjects"
    )
    calibration_subjects = _positive_int(
        data.get("calibration_subjects", 0), "data.calibration_subjects"
    )
    if data_ints["n_subjects"] < final_holdout + calibration_subjects + 3:
        raise ValueError(
            "data.n_subjects must cover final_holdout_subjects + calibration_subjects + 3 "
            "participant-disjoint roles"
        )
    for key in ("c1_max_iter", "c2_epochs", "c3_epochs", "batch_size", "hidden_dim"):
        _positive_int(models.get(key, 0), f"models.{key}")
    if _finite_float(models.get("learning_rate", 0), "models.learning_rate") <= 0:
        raise ValueError("models.learning_rate must be positive")
    if not 0 <= _finite_float(models.get("dropout", 0), "models.dropout") < 1:
        raise ValueError("models.dropout must be in [0, 1)")
    device = str(models.get("device", "cpu")).lower()
    if device not in {"cpu", "cuda", "auto"}:
        raise ValueError("models.device must be one of: cpu, cuda, auto")
    grid = evaluation.get("threshold_grid")
    if not isinstance(grid, list) or not grid:
        raise ValueError("evaluation.threshold_grid must be a non-empty list")
    parsed_grid = [_finite_float(value, "evaluation.threshold_grid entry") for value in grid]
    if any(not 0 < value < 1 for value in parsed_grid):
        raise ValueError("evaluation.threshold_grid values must lie strictly between zero and one")
    if sorted(parsed_grid) != parsed_grid:
        raise ValueError("evaluation.threshold_grid must be sorted ascending")
    if len(set(parsed_grid)) != len(parsed_grid):
        raise ValueError("evaluation.threshold_grid must not contain duplicate thresholds")
    if (
        _finite_float(
            evaluation.get("target_false_alert_minutes_per_hour", -1),
            "evaluation.target_false_alert_minutes_per_hour",
        )
        < 0
    ):
        raise ValueError("evaluation.target_false_alert_minutes_per_hour must be non-negative")
    if _positive_int(evaluation.get("ece_bins", 0), "evaluation.ece_bins") < 2:
        raise ValueError("evaluation.ece_bins must be at least two")
    _positive_int(evaluation.get("bootstrap_resamples", 0), "evaluation.bootstrap_resamples")
    gates = evaluation.get("release_gates")
    if not isinstance(gates, dict):
        raise ValueError("evaluation.release_gates must be a mapping")
    required_gates = {
        "min_episode_recall_uplift_pp",
        "min_bootstrap_subjects",
        "max_macro_f1_decline_pp",
        "max_ece_regression",
        "max_worst_subject_recall_decline_pp",
        "max_relative_fault_degradation",
        "max_inference_p95_ms",
    }
    missing_gates = sorted(required_gates - set(gates))
    if missing_gates:
        raise ValueError(f"evaluation.release_gates missing required keys: {missing_gates}")
    for key in required_gates:
        _finite_float(gates[key], f"evaluation.release_gates.{key}")
    _positive_int(
        gates["min_bootstrap_subjects"], "evaluation.release_gates.min_bootstrap_subjects"
    )
    if (
        _finite_float(
            gates["max_relative_fault_degradation"],
            "evaluation.release_gates.max_relative_fault_degradation",
        )
        < 0
    ):
        raise ValueError(
            "evaluation.release_gates.max_relative_fault_degradation must be non-negative"
        )
    if (
        _finite_float(gates["max_ece_regression"], "evaluation.release_gates.max_ece_regression")
        < 0
    ):
        raise ValueError("evaluation.release_gates.max_ece_regression must be non-negative")
    if (
        _finite_float(
            gates["max_inference_p95_ms"], "evaluation.release_gates.max_inference_p95_ms"
        )
        <= 0
    ):
        raise ValueError("evaluation.release_gates.max_inference_p95_ms must be positive")
    _nonnegative_int(policy.get("elevated_class", -1), "policy.elevated_class")
    for key in ("confidence_threshold", "quality_threshold"):
        value = _finite_float(policy.get(key, -1), f"policy.{key}")
        if not 0 <= value <= 1:
            raise ValueError(f"policy.{key} must be in [0, 1]")


def _hashable_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        config, descriptor = _default_config()
    else:
        chosen = Path(path)
        config = _read_yaml(chosen)
        descriptor = str(chosen)
    validate_config(config)
    config["_config_path"] = descriptor
    config["_config_hash"] = stable_json_hash(_hashable_config(config))
    return config


def with_overrides(config: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    copied = deepcopy(config)
    for dotted_key, value in overrides.items():
        cursor = copied
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    validate_config(copied)
    copied["_config_hash"] = stable_json_hash(_hashable_config(copied))
    return copied
