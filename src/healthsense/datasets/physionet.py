from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from healthsense.datasets.contracts import CanonicalBatch
from healthsense.datasets.wesad import _availability_flatline_quality, _resample_segment
from healthsense.utils.io import file_fingerprint, write_json, write_table

E4_RATES = {"BVP": 64, "EDA": 4, "TEMP": 4, "ACC": 32}


def _read_empatica_csv(path: Path) -> tuple[float, float, np.ndarray]:
    """Read standard Empatica E4 raw CSV: UTC start row, sampling-rate row, then samples."""
    raw = pd.read_csv(path, header=None)
    if len(raw) < 3:
        raise ValueError(f"too few rows in {path}")
    start_utc = float(raw.iloc[0, 0])
    sample_rate = float(raw.iloc[1, 0])
    samples = raw.iloc[2:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.shape[1] == 1:
        samples = samples.reshape(-1, 1)
    return start_utc, sample_rate, samples


def _load_tag_intervals(tag_map_path: str | Path | None) -> pd.DataFrame:
    if tag_map_path is None:
        return pd.DataFrame(
            columns=[
                "session_family",
                "subject_id",
                "start_tag_index",
                "end_tag_index",
                "state_label",
            ]
        )
    table = pd.read_csv(tag_map_path)
    required = {"session_family", "subject_id", "start_tag_index", "end_tag_index", "state_label"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"tag map missing columns {sorted(missing)}")
    return table


def inspect_physionet(source_root: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Inspect official PhysioNet download structure and source constraints."""
    root = Path(source_root)
    if not root.exists():
        raise FileNotFoundError(root)
    csvs = sorted(root.rglob("*.csv"))
    txts = sorted(root.rglob("*.txt"))
    sessions = sorted({str(p.parent.relative_to(root)) for p in csvs})
    known_signal_tokens = ["BVP", "EDA", "TEMP", "ACC", "HR", "IBI", "tag"]
    signal_counts = {
        token: sum(token.lower() in p.name.lower() for p in csvs) for token in known_signal_tokens
    }
    manifest = {
        "dataset": "PhysioNet Wearable Device Dataset from Induced Stress and Structured Exercise Sessions",
        "source_root": str(root.resolve()),
        "csv_count": len(csvs),
        "txt_count": len(txts),
        "session_directories": sessions[:200],
        "signal_file_counts": signal_counts,
        "sample_files": [file_fingerprint(p) for p in csvs[:30]],
        "conversion_rule": "Protocol tags may create labels only; they must not be model features.",
        "confound_rule": "Aerobic and anaerobic sessions stay separate from stress/rest target labels.",
        "required_tag_map_schema": [
            "session_family",
            "subject_id",
            "start_tag_index",
            "end_tag_index",
            "state_label",
        ],
    }
    write_json(output_path, manifest)
    return manifest


def prepare_physionet(
    source_root: str | Path,
    output_dir: str | Path,
    *,
    tag_map_path: str | Path | None = None,
    window_seconds: int = 30,
    hop_seconds: int = 5,
    sequence_length: int = 60,
) -> dict[str, Any]:
    """Convert standard Empatica E4 folders into canonical windows.

    A verified protocol tag map is deliberately required to attach stress/rest labels. Without it,
    windows remain `state_label=-1` and cannot enter the supervised external evaluation path.
    """
    source = Path(source_root)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag_map = _load_tag_intervals(tag_map_path)
    session_dirs = sorted({p.parent for p in source.rglob("BVP.csv")})
    if not session_dirs:
        raise FileNotFoundError("No BVP.csv files found under source root")
    rows: list[dict[str, Any]] = []
    seqs: list[np.ndarray] = []
    labels: list[int] = []
    masks: list[np.ndarray] = []
    qualities: list[np.ndarray] = []
    lags: list[np.ndarray] = []
    motion_scores: list[float] = []
    channel_modalities: list[int] | None = None

    for directory in session_dirs:
        available: dict[str, tuple[float, float, np.ndarray]] = {}
        for modality in E4_RATES:
            candidate = directory / f"{modality}.csv"
            if candidate.exists():
                available[modality] = _read_empatica_csv(candidate)
        if "BVP" not in available:
            continue
        relative = directory.relative_to(source)
        parts = relative.parts
        session_family = parts[0].upper() if parts else "UNKNOWN"
        source_subject_id = parts[-1]
        # Preserve global uniqueness across session-family folders. The raw source identifier is
        # retained separately for audited tag-map joins.
        subject_id = f"{session_family}::{source_subject_id}"
        session_id = "::".join(parts)
        start_utc = available["BVP"][0]
        duration_s = min(len(payload[2]) / payload[1] for payload in available.values())
        tags_path = directory / "tags.csv"
        tag_times = np.array([], dtype=float)
        if tags_path.exists():
            tag_values = pd.read_csv(tags_path, header=None).iloc[:, 0]
            tag_times = pd.to_numeric(tag_values, errors="coerce").dropna().to_numpy(dtype=float)
        subject_map = tag_map[
            (tag_map["session_family"].astype(str).str.upper() == session_family)
            & (tag_map["subject_id"].astype(str) == source_subject_id)
        ]
        previous_label: int | None = None
        for start_s in np.arange(0, max(0, duration_s - window_seconds) + 1e-9, hop_seconds):
            channels: list[np.ndarray] = []
            q: list[float] = []
            mask: list[float] = []
            motion = 0.0
            for mi, modality in enumerate(E4_RATES):
                if modality not in available:
                    # preserve a stable four-modality channel layout: 1 BVP, 1 EDA, 1 TEMP, 3 ACC
                    expected = 3 if modality == "ACC" else 1
                    channels.append(np.zeros((sequence_length, expected), dtype=np.float32))
                    q.append(0.0)
                    mask.append(0.0)
                    continue
                _, rate, data = available[modality]
                left, right = int(start_s * rate), int((start_s + window_seconds) * rate)
                segment = data[left:right]
                expected = 3 if modality == "ACC" else 1
                if len(segment) == 0:
                    segment = np.zeros((1, expected), dtype=np.float32)
                reduced = _resample_segment(segment, sequence_length)
                if reduced.shape[1] != expected:
                    reduced = reduced[:, :expected]
                    if reduced.shape[1] < expected:
                        reduced = np.pad(reduced, ((0, 0), (0, expected - reduced.shape[1])))
                channels.append(reduced)
                q.append(_availability_flatline_quality(reduced))
                mask.append(1.0)
                if modality == "ACC":
                    motion = float(np.clip(np.std(reduced) / 5.0, 0, 1))
            sequence = np.concatenate(channels, axis=1).astype(np.float32)
            if channel_modalities is None:
                channel_modalities = [0, 1, 2, 3, 3, 3]
            # Label construction requires a reviewed tag mapping; tags themselves are never features.
            label = -1
            if len(subject_map) and len(tag_times):
                window_start = start_utc + start_s
                window_end = window_start + window_seconds
                for _, mapping in subject_map.iterrows():
                    a, b = int(mapping["start_tag_index"]), int(mapping["end_tag_index"])
                    if 0 <= a < len(tag_times) and 0 <= b < len(tag_times):
                        lo, hi = sorted((tag_times[a], tag_times[b]))
                        # Do not assign a protocol label to a window that straddles a boundary.
                        if lo <= window_start and window_end <= hi:
                            label = int(mapping["state_label"])
                            break
            seqs.append(sequence)
            labels.append(label)
            masks.append(np.asarray(mask, dtype=np.float32))
            qualities.append(np.asarray(q, dtype=np.float32))
            lags.append(np.zeros(4, dtype=np.float32))
            motion_scores.append(motion)
            rows.append(
                {
                    "dataset_id": "physionet_wearable_device",
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "sensor_config": "wrist_e4",
                    "feature_representation": "native",
                    "feature_schema_id": "e4_wrist_v1",
                    "window_end_s": float(start_s + window_seconds),
                    "state_label": label,
                    "transition_label": int(
                        label >= 0 and previous_label is not None and previous_label != label
                    ),
                    "split_role": "unassigned",
                    "session_family": session_family,
                    "source_subject_id": source_subject_id,
                }
            )
            if label >= 0:
                previous_label = label
    if not seqs:
        raise ValueError("No windows produced")
    batch = CanonicalBatch(
        sequence=np.stack(seqs),
        labels=np.asarray(labels, dtype=np.int64),
        metadata=pd.DataFrame(rows),
        modality_mask=np.stack(masks),
        quality=np.stack(qualities),
        timing_offset_ms=np.stack(lags),
        motion_artifact_score=np.asarray(motion_scores, dtype=np.float32),
        channel_modalities=np.asarray(channel_modalities, dtype=np.int64),
    )
    # Unknown labels are valid for an auditable prepared artifact but blocked by every supervised
    # evaluation entrypoint. Validate the complete source contract in both cases.
    batch.validate()
    batch_for_storage = batch
    np.savez_compressed(
        out / "physionet_canonical_windows.npz",
        sequence=batch_for_storage.sequence,
        labels=batch_for_storage.labels,
        modality_mask=batch_for_storage.modality_mask,
        quality=batch_for_storage.quality,
        timing_offset_ms=batch_for_storage.timing_offset_ms,
        motion_artifact_score=batch_for_storage.motion_artifact_score,
        channel_modalities=batch_for_storage.channel_modalities,
    )
    write_table(batch_for_storage.metadata, out / "physionet_window_metadata.csv")
    manifest = {
        "dataset": "PhysioNet Wearable Device Dataset from Induced Stress and Structured Exercise Sessions",
        "source_root": str(source.resolve()),
        "n_windows": int(len(rows)),
        "labeled_windows": int(np.sum(np.asarray(labels) >= 0)),
        "unlabeled_windows": int(np.sum(np.asarray(labels) < 0)),
        "tag_map_path": str(tag_map_path) if tag_map_path else None,
        "feature_schema_id": "e4_wrist_v1",
        "canonical_modalities": ["BVP", "EDA", "TEMP", "ACC"],
        "label_rule": "Tag map only; raw tags are not model features.",
        "exercise_rule": "Exercise windows must be reported as a separate confound/OOD challenge, not merged into stress/rest.",
        "quality_proxy": "availability_flatline_proxy; not a validated physiological SQI",
        "tag_map_fingerprint": file_fingerprint(tag_map_path) if tag_map_path else None,
        "source_files": [file_fingerprint(p) for p in sorted(source.rglob("*.csv"))],
    }
    write_json(out / "physionet_source_manifest.json", manifest)
    return manifest
