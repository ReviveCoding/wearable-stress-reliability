from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from healthsense.datasets.contracts import CanonicalBatch


@dataclass(frozen=True)
class QualitySummary:
    n_samples: int
    n_subjects: int
    missingness_rate: float
    mean_quality: float
    high_motion_rate: float
    invalid_count: int


def assess_batch_quality(batch: CanonicalBatch) -> QualitySummary:
    batch.validate()
    return QualitySummary(
        n_samples=batch.n_samples,
        n_subjects=int(batch.metadata["subject_id"].nunique()),
        missingness_rate=float(1.0 - batch.modality_mask.mean()),
        mean_quality=float(batch.quality.mean()),
        high_motion_rate=float((batch.motion_artifact_score >= 0.7).mean()),
        invalid_count=0,
    )


def quality_dict(batch: CanonicalBatch) -> dict[str, float | int]:
    return asdict(assess_batch_quality(batch))


def apply_modality_dropout(batch: CanonicalBatch, modality: int) -> CanonicalBatch:
    if modality < 0 or modality >= batch.modality_mask.shape[1]:
        raise ValueError("invalid modality index")
    out = batch.subset(np.arange(batch.n_samples))
    out.modality_mask[:, modality] = 0.0
    out.quality[:, modality] = 0.0
    channels = out.channel_modalities == modality
    out.sequence[:, :, channels] = 0.0
    out.validate()
    return out


def inject_timing_lag(batch: CanonicalBatch, modality: int, lag_ms: float) -> CanonicalBatch:
    out = batch.subset(np.arange(batch.n_samples))
    out.timing_offset_ms[:, modality] += lag_ms
    # deterministic coarse time-shift; strictly within each causal window.
    shift = max(1, int(round(abs(lag_ms) / 1000.0)))
    channels = out.channel_modalities == modality
    if lag_ms > 0:
        out.sequence[:, shift:, channels] = out.sequence[:, :-shift, channels]
        out.sequence[:, :shift, channels] = 0.0
    else:
        out.sequence[:, :-shift, channels] = out.sequence[:, shift:, channels]
        out.sequence[:, -shift:, channels] = 0.0
    out.validate()
    return out


def inject_motion_noise(batch: CanonicalBatch, scale: float = 0.4, seed: int = 0) -> CanonicalBatch:
    out = batch.subset(np.arange(batch.n_samples))
    rng = np.random.default_rng(seed)
    channels = np.isin(out.channel_modalities, [0, 1])
    modifier = out.motion_artifact_score[:, None, None]
    out.sequence[:, :, channels] += (
        rng.normal(0, scale, size=out.sequence[:, :, channels].shape) * modifier
    )
    out.motion_artifact_score = np.clip(out.motion_artifact_score + 0.2, 0, 1)
    out.quality = np.clip(out.quality - 0.15, 0, 1)
    out.validate()
    return out
