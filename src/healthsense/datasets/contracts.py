from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_METADATA = {
    "dataset_id",
    "subject_id",
    "session_id",
    "sensor_config",
    "window_end_s",
    "state_label",
    "transition_label",
    "split_role",
}
UNKNOWN_LABEL = -1


@dataclass(frozen=True)
class WindowRecord:
    dataset_id: str
    subject_id: str
    session_id: str
    sensor_config: str
    window_end_s: float
    state_label: int
    transition_label: int
    split_role: str = "unassigned"


@dataclass
class CanonicalBatch:
    """Causal, fixed-length multi-modal windows and auditable metadata.

    ``sequence`` is ``[N, T, C]`` and masks/quality/timing are ``[N, M]``. Channels map to
    modality ids through ``channel_modalities``. Metadata is audit-only and must never become a
    predictive feature. ``state_label=-1`` is allowed only for *prepared but unreviewed source
    artifacts*; every supervised evaluation entrypoint must reject it before training.
    """

    sequence: np.ndarray
    labels: np.ndarray
    metadata: pd.DataFrame
    modality_mask: np.ndarray
    quality: np.ndarray
    timing_offset_ms: np.ndarray
    motion_artifact_score: np.ndarray
    channel_modalities: np.ndarray

    def validate(self) -> None:
        n = self.sequence.shape[0]
        if n < 1:
            raise ValueError("canonical batch must contain at least one window")
        if self.sequence.ndim != 3:
            raise ValueError("sequence must have shape [N, T, C]")
        if self.labels.shape != (n,):
            raise ValueError("labels must have shape [N]")
        if len(self.metadata) != n:
            raise ValueError("metadata rows must equal N")
        if (
            self.modality_mask.ndim != 2
            or self.quality.ndim != 2
            or self.timing_offset_ms.ndim != 2
        ):
            raise ValueError("modality_mask, quality, and timing_offset_ms must have shape [N, M]")
        if self.modality_mask.shape[0] != n or self.quality.shape[0] != n:
            raise ValueError("mask/quality first dimensions must equal N")
        if (
            self.modality_mask.shape != self.quality.shape
            or self.modality_mask.shape != self.timing_offset_ms.shape
        ):
            raise ValueError("modality_mask, quality, and timing_offset_ms must share shape [N, M]")
        if self.motion_artifact_score.shape != (n,):
            raise ValueError("motion_artifact_score must have shape [N]")
        if (
            self.channel_modalities.ndim != 1
            or len(self.channel_modalities) != self.sequence.shape[2]
        ):
            raise ValueError("channel_modalities must map every sequence channel")
        if not np.issubdtype(self.channel_modalities.dtype, np.integer):
            raise ValueError("channel_modalities must use an integer dtype")
        if self.sequence.shape[1] < 1 or self.sequence.shape[2] < 1:
            raise ValueError("sequence must contain at least one timestep and one channel")
        if np.any(self.channel_modalities < 0) or np.any(
            self.channel_modalities >= self.modality_mask.shape[1]
        ):
            raise ValueError("channel_modalities contains invalid modality ids")
        if not REQUIRED_METADATA.issubset(set(self.metadata.columns)):
            missing = sorted(REQUIRED_METADATA - set(self.metadata.columns))
            raise ValueError(f"metadata missing required fields: {missing}")
        identity_cols = ["dataset_id", "subject_id", "session_id", "sensor_config", "window_end_s"]
        if self.metadata[identity_cols].isna().any().any():
            raise ValueError("canonical window identity fields must not be null")
        if any(
            (self.metadata[column].astype(str).str.strip() == "").any()
            for column in identity_cols[:-1]
        ):
            raise ValueError("canonical window identity text fields must not be empty")
        if self.metadata.duplicated(identity_cols).any():
            raise ValueError("duplicate canonical window identity detected")
        if np.any(~np.isfinite(self.sequence)):
            raise ValueError("sequence contains non-finite values")
        if (
            np.any(~np.isfinite(self.modality_mask))
            or np.any(~np.isfinite(self.quality))
            or np.any(~np.isfinite(self.timing_offset_ms))
            or np.any(~np.isfinite(self.motion_artifact_score))
        ):
            raise ValueError("mask, quality, timing, and motion values must be finite")
        if not np.isfinite(self.metadata["window_end_s"].to_numpy(dtype=float)).all():
            raise ValueError("window_end_s must be finite")
        if np.any((self.modality_mask < 0) | (self.modality_mask > 1)):
            raise ValueError("modality_mask must be in [0,1]")
        if not np.isin(self.modality_mask, [0.0, 1.0]).all():
            raise ValueError("modality_mask must use binary 0/1 availability values")
        if np.any((self.quality < 0) | (self.quality > 1)):
            raise ValueError("quality must be in [0,1]")
        if np.any((self.motion_artifact_score < 0) | (self.motion_artifact_score > 1)):
            raise ValueError("motion_artifact_score must be in [0,1]")
        if not np.issubdtype(self.labels.dtype, np.integer):
            raise ValueError("labels must use an integer dtype")
        if np.any(self.labels < UNKNOWN_LABEL):
            raise ValueError("labels may only use non-negative class ids or -1 for unknown")
        try:
            metadata_labels = pd.to_numeric(self.metadata["state_label"], errors="raise").to_numpy(
                dtype=float
            )
        except Exception as exc:
            raise ValueError("metadata state_label must be numeric") from exc
        if np.any(~np.isfinite(metadata_labels)) or not np.allclose(
            metadata_labels, np.round(metadata_labels), atol=0.0
        ):
            raise ValueError("metadata state_label must contain finite integer values")
        if not np.array_equal(metadata_labels.astype(np.int64), self.labels.astype(np.int64)):
            raise ValueError("metadata state_label must exactly match labels")
        transitions = self.metadata["transition_label"].to_numpy()
        if not np.isin(transitions, [0, 1]).all():
            raise ValueError("metadata transition_label must be binary")

    @property
    def has_unknown_labels(self) -> bool:
        return bool(np.any(self.labels == UNKNOWN_LABEL))

    @property
    def n_samples(self) -> int:
        return int(self.sequence.shape[0])

    @property
    def n_classes(self) -> int:
        if self.has_unknown_labels:
            raise ValueError(
                "n_classes is undefined for batches with unknown labels; attach a reviewed label map "
                "or filter unknown windows before supervised training"
            )
        return int(np.max(self.labels) + 1)

    def subset(self, indices: np.ndarray | list[int]) -> "CanonicalBatch":
        idx = np.asarray(indices, dtype=int)
        return CanonicalBatch(
            sequence=self.sequence[idx].copy(),
            labels=self.labels[idx].copy(),
            metadata=self.metadata.iloc[idx].reset_index(drop=True).copy(),
            modality_mask=self.modality_mask[idx].copy(),
            quality=self.quality[idx].copy(),
            timing_offset_ms=self.timing_offset_ms[idx].copy(),
            motion_artifact_score=self.motion_artifact_score[idx].copy(),
            channel_modalities=self.channel_modalities.copy(),
        )

    def as_feature_frame(self) -> pd.DataFrame:
        """Causal tabular aggregation for C1 without metadata or label-derived leakage."""
        n, _, channels = self.sequence.shape
        rows: dict[str, np.ndarray] = {}
        for c in range(channels):
            x = self.sequence[:, :, c]
            rows[f"ch{c}_mean"] = x.mean(axis=1)
            rows[f"ch{c}_std"] = x.std(axis=1)
            rows[f"ch{c}_min"] = x.min(axis=1)
            rows[f"ch{c}_max"] = x.max(axis=1)
            rows[f"ch{c}_slope"] = x[:, -1] - x[:, 0]
        for m in range(self.modality_mask.shape[1]):
            rows[f"mask_{m}"] = self.modality_mask[:, m]
            rows[f"quality_{m}"] = self.quality[:, m]
            rows[f"timing_{m}"] = self.timing_offset_ms[:, m]
        rows["motion_artifact_score"] = self.motion_artifact_score
        return pd.DataFrame(rows, index=np.arange(n))

    def fingerprint_payload(self) -> dict[str, Any]:
        values, counts = np.unique(self.labels, return_counts=True)
        return {
            "n_samples": self.n_samples,
            "sequence_shape": list(self.sequence.shape),
            "label_counts": {str(int(k)): int(v) for k, v in zip(values, counts, strict=True)},
            "unknown_label_count": int((self.labels == UNKNOWN_LABEL).sum()),
            "subjects": sorted(self.metadata["subject_id"].astype(str).unique().tolist()),
            "datasets": sorted(self.metadata["dataset_id"].astype(str).unique().tolist()),
            "feature_schema_ids": sorted(
                self.metadata.get("feature_schema_id", pd.Series(["unspecified"]))
                .astype(str)
                .unique()
                .tolist()
            ),
        }
