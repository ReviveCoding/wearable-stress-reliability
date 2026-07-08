from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from healthsense.datasets.contracts import CanonicalBatch
from healthsense.utils.io import ensure_dir, file_fingerprint, write_json, write_table

# WESAD source layout documented by the original dataset: signal/wrist and signal/chest plus labels.
# The wrist order intentionally matches the PhysioNet E4 canonical schema: BVP, EDA, TEMP, ACC(x3).
WESAD_RATES = {
    "wrist": {"BVP": 64, "EDA": 4, "TEMP": 4, "ACC": 32},
    "chest": {"ACC": 700, "ECG": 700, "EDA": 700, "EMG": 700, "Resp": 700, "Temp": 700},
}
WESAD_LABEL_MAP = {1: 0, 2: 1, 3: 2}  # baseline -> 0, stress -> 1, amusement -> 2

# Each tuple is ``(source_signal_name, canonical_modality_name)``. The shared representation enables
# chest-to-wrist analysis without pretending the non-shared channels are directly comparable.
WESAD_REPRESENTATIONS: dict[str, dict[str, tuple[tuple[str, str], ...]]] = {
    "native": {
        "wrist": (("BVP", "BVP"), ("EDA", "EDA"), ("TEMP", "TEMP"), ("ACC", "ACC")),
        "chest": (
            ("ACC", "ACC"),
            ("ECG", "ECG"),
            ("EDA", "EDA"),
            ("EMG", "EMG"),
            ("Resp", "Resp"),
            ("Temp", "TEMP"),
        ),
    },
    "shared_eda_temp_acc": {
        "wrist": (("EDA", "EDA"), ("TEMP", "TEMP"), ("ACC", "ACC")),
        "chest": (("EDA", "EDA"), ("Temp", "TEMP"), ("ACC", "ACC")),
    },
}


def _resample_segment(x: np.ndarray, target_steps: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    if len(x) == target_steps:
        return x
    old = np.linspace(0, 1, len(x))
    new = np.linspace(0, 1, target_steps)
    return np.stack([np.interp(new, old, x[:, i]) for i in range(x.shape[1])], axis=1).astype(
        np.float32
    )


def _window_label_purity(labels: np.ndarray) -> tuple[int | None, float]:
    labels = labels.astype(int)
    valid = labels[np.isin(labels, list(WESAD_LABEL_MAP))]
    if len(valid) == 0:
        return None, 0.0
    vals, counts = np.unique(valid, return_counts=True)
    winner = int(vals[np.argmax(counts)])
    return winner, float(counts.max() / len(valid))


def _availability_flatline_quality(segment: np.ndarray) -> float:
    """Conservative non-clinical data-availability / flatline proxy.

    This is intentionally not described as a validated physiological signal-quality index. It is
    only a deterministic gate for missing, non-finite, or nearly constant windows.
    """
    x = np.asarray(segment, dtype=np.float32)
    finite = np.isfinite(x)
    finite_fraction = float(finite.mean())
    if finite_fraction == 0.0:
        return 0.0
    clean = np.where(finite, x, 0.0)
    if clean.shape[0] < 2:
        return float(np.clip(finite_fraction * 0.25, 0.0, 1.0))
    diff = np.abs(np.diff(clean, axis=0))
    scale = np.median(np.abs(clean - np.median(clean, axis=0, keepdims=True))) + 1e-6
    flatline_fraction = float((diff <= 1e-3 * scale).mean())
    return float(np.clip(finite_fraction * (1.0 - 0.75 * flatline_fraction), 0.0, 1.0))


def _representation_spec(sensor_config: str, representation: str) -> tuple[tuple[str, str], ...]:
    if sensor_config not in WESAD_RATES:
        raise ValueError(f"sensor_config must be one of {sorted(WESAD_RATES)}")
    if representation not in WESAD_REPRESENTATIONS:
        raise ValueError(f"representation must be one of {sorted(WESAD_REPRESENTATIONS)}")
    return WESAD_REPRESENTATIONS[representation][sensor_config]


def _feature_schema_id(sensor_config: str, representation: str) -> str:
    if representation == "native" and sensor_config == "wrist":
        # Explicitly align with the PhysioNet E4 adapter: BVP, EDA, TEMP, ACC(x3).
        return "e4_wrist_v1"
    if representation == "shared_eda_temp_acc":
        return "wesad_shared_eda_temp_acc_v1"
    return f"wesad_{sensor_config}_{representation}_v1"


def _output_names(sensor_config: str, representation: str) -> tuple[str, str]:
    if representation == "native":
        return "wesad_canonical_windows.npz", "wesad_window_metadata.csv"
    stem = f"wesad_{sensor_config}_{representation}_canonical_windows"
    return f"{stem}.npz", f"{stem}_metadata.csv"


def prepare_wesad(
    source_root: str | Path,
    output_dir: str | Path,
    *,
    trust_pickle: bool,
    sensor_config: str = "wrist",
    representation: str = "native",
    window_seconds: int = 30,
    hop_seconds: int = 5,
    sequence_length: int = 60,
    min_label_purity: float = 0.90,
) -> dict[str, Any]:
    """Convert official WESAD subject pickle files to canonical window artifacts.

    ``representation='shared_eda_temp_acc'`` creates a conservative common feature layout for
    WESAD chest-to-wrist transfer. It is not a device-equivalence claim. Pickle is unsafe for
    untrusted data, so callers must explicitly acknowledge trusted, hash-verified source files.
    """
    if not trust_pickle:
        raise ValueError(
            "WESAD uses pickle; pass trust_pickle=True only for official, hash-verified source files."
        )
    source = Path(source_root)
    out = ensure_dir(output_dir)
    if not 0.5 <= min_label_purity <= 1.0:
        raise ValueError("min_label_purity must be within [0.5, 1.0]")
    if window_seconds <= 0 or hop_seconds <= 0 or sequence_length <= 0:
        raise ValueError("window_seconds, hop_seconds, and sequence_length must be positive")
    pkl_files = sorted(source.glob("S*/S*.pkl"))
    if not pkl_files:
        raise FileNotFoundError("Expected WESAD subject pickles like <source_root>/S2/S2.pkl")

    spec = _representation_spec(sensor_config, representation)
    source_modalities = [source_name for source_name, _ in spec]
    canonical_modalities = [canonical_name for _, canonical_name in spec]
    rates = WESAD_RATES[sensor_config]
    feature_schema_id = _feature_schema_id(sensor_config, representation)

    all_rows: list[dict[str, Any]] = []
    all_seq: list[np.ndarray] = []
    all_labels: list[int] = []
    all_masks: list[np.ndarray] = []
    all_quality: list[np.ndarray] = []
    all_timing: list[np.ndarray] = []
    all_motion: list[float] = []
    channel_modalities: list[int] = []

    for pkl_path in pkl_files:
        with pkl_path.open("rb") as f:
            payload = pickle.load(f, encoding="latin1")  # nosec B301: explicit user acknowledgement above
        try:
            signals = payload["signal"][sensor_config]
            labels = np.asarray(payload["label"])
        except Exception as exc:
            raise ValueError(f"Unexpected WESAD payload in {pkl_path}: {exc}") from exc
        missing = [name for name in source_modalities if name not in signals]
        if missing:
            raise ValueError(
                f"WESAD {sensor_config} payload in {pkl_path} is missing required signals {missing}"
            )
        usable_seconds = min(
            len(labels) / 700.0,
            *(len(np.asarray(signals[name])) / rates[name] for name in source_modalities),
        )
        previous_mapped_label: int | None = None
        for start_s in np.arange(0, max(0, usable_seconds - window_seconds) + 1e-9, hop_seconds):
            label_slice = labels[int(start_s * 700) : int((start_s + window_seconds) * 700)]
            raw_label, label_purity = _window_label_purity(label_slice)
            if raw_label is None or label_purity < min_label_purity:
                continue
            channels: list[np.ndarray] = []
            modality_quality: list[float] = []
            motion = 0.0
            for source_name, canonical_name in spec:
                arr = np.asarray(signals[source_name])
                start = int(start_s * rates[source_name])
                end = int((start_s + window_seconds) * rates[source_name])
                segment = arr[start:end]
                if len(segment) == 0:
                    break
                reduced = _resample_segment(segment, sequence_length)
                if reduced.ndim == 1:
                    reduced = reduced[:, None]
                channels.append(reduced)
                modality_quality.append(_availability_flatline_quality(reduced))
                if canonical_name == "ACC":
                    motion = float(np.clip(np.std(reduced), 0.0, 5.0) / 5.0)
            else:
                mapped_label = WESAD_LABEL_MAP[raw_label]
                all_seq.append(np.concatenate(channels, axis=1).astype(np.float32))
                all_labels.append(mapped_label)
                all_masks.append(np.ones(len(spec), dtype=np.float32))
                all_quality.append(np.asarray(modality_quality, dtype=np.float32))
                all_timing.append(np.zeros(len(spec), dtype=np.float32))
                all_motion.append(motion)
                subject = pkl_path.parent.name
                all_rows.append(
                    {
                        "dataset_id": "wesad",
                        "subject_id": subject,
                        "session_id": f"{subject}_lab",
                        "sensor_config": sensor_config,
                        "feature_representation": representation,
                        "feature_schema_id": feature_schema_id,
                        "window_end_s": float(start_s + window_seconds),
                        "state_label": mapped_label,
                        "transition_label": int(
                            previous_mapped_label is not None
                            and previous_mapped_label != mapped_label
                        ),
                        "label_purity": float(label_purity),
                        "split_role": "unassigned",
                    }
                )
                previous_mapped_label = mapped_label
        if not channel_modalities:
            for modality_idx, source_name in enumerate(source_modalities):
                raw = np.asarray(signals[source_name])
                n_channels = raw.shape[1] if raw.ndim > 1 else 1
                channel_modalities.extend([modality_idx] * n_channels)

    if not all_seq:
        raise ValueError(
            "No usable WESAD windows produced. Check source structure and window configuration."
        )
    batch = CanonicalBatch(
        sequence=np.stack(all_seq),
        labels=np.asarray(all_labels, dtype=np.int64),
        metadata=pd.DataFrame(all_rows),
        modality_mask=np.stack(all_masks),
        quality=np.stack(all_quality),
        timing_offset_ms=np.stack(all_timing),
        motion_artifact_score=np.asarray(all_motion, dtype=np.float32),
        channel_modalities=np.asarray(channel_modalities, dtype=np.int64),
    )
    batch.validate()
    npz_name, metadata_name = _output_names(sensor_config, representation)
    np.savez_compressed(
        out / npz_name,
        sequence=batch.sequence,
        labels=batch.labels,
        modality_mask=batch.modality_mask,
        quality=batch.quality,
        timing_offset_ms=batch.timing_offset_ms,
        motion_artifact_score=batch.motion_artifact_score,
        channel_modalities=batch.channel_modalities,
    )
    write_table(batch.metadata, out / metadata_name)
    manifest = {
        "dataset": "WESAD",
        "source_root": str(source.resolve()),
        "source_files": [file_fingerprint(p) for p in pkl_files],
        "adapter": "healthsense.datasets.wesad.prepare_wesad",
        "sensor_config": sensor_config,
        "feature_representation": representation,
        "feature_schema_id": feature_schema_id,
        "canonical_modalities": canonical_modalities,
        "output_npz": npz_name,
        "output_metadata": metadata_name,
        "window_seconds": window_seconds,
        "hop_seconds": hop_seconds,
        "sequence_length": sequence_length,
        "min_label_purity": min_label_purity,
        "quality_proxy": "availability_flatline_proxy; not a validated physiological SQI",
        "batch": batch.fingerprint_payload(),
        "safety": "Only use official, hash-verified pickles; raw data are not redistributed.",
    }
    write_json(out / "wesad_source_manifest.json", manifest)
    return manifest
