from __future__ import annotations

import numpy as np
import pandas as pd

from healthsense.datasets.contracts import CanonicalBatch


def _subject_state_sequence(rng: np.random.Generator, windows_per_subject: int) -> np.ndarray:
    """Generate variable, repeated state episodes without a shared fixed window-position shortcut.

    This remains a deterministic engineering fixture, not a physiological simulator. Every subject
    sees all three proxy states, but state order and episode lengths vary by subject so sequence
    models cannot exploit a common protocol index alone.
    """
    if windows_per_subject < 6:
        raise ValueError("windows_per_subject must be at least 6 for all proxy states")
    if windows_per_subject >= 12:
        patterns = (
            np.array([0, 1, 0, 2], dtype=np.int64),
            np.array([0, 2, 0, 1], dtype=np.int64),
            np.array([1, 0, 2, 0], dtype=np.int64),
            np.array([2, 0, 1, 0], dtype=np.int64),
        )
        pattern = patterns[int(rng.integers(0, len(patterns)))]
    else:
        patterns = (
            np.array([0, 1, 2], dtype=np.int64),
            np.array([1, 0, 2], dtype=np.int64),
            np.array([2, 0, 1], dtype=np.int64),
        )
        pattern = patterns[int(rng.integers(0, len(patterns)))]
    minimum = 2
    remaining = windows_per_subject - minimum * len(pattern)
    extra = rng.multinomial(remaining, rng.dirichlet(np.ones(len(pattern))))
    return np.concatenate(
        [
            np.repeat(state, minimum + int(count))
            for state, count in zip(pattern, extra, strict=True)
        ]
    )


def make_synthetic_fixture(
    seed: int = 17,
    n_subjects: int = 9,
    windows_per_subject: int = 24,
    sequence_length: int = 20,
    n_modalities: int = 4,
    n_channels: int = 8,
) -> CanonicalBatch:
    """Creates a deterministic, public-style fixture, not a physiological simulator.

    Classes: 0 baseline, 1 induced-stress proxy, 2 amusement proxy. Signals deliberately
    include subject offsets, quality failures, asynchronous modalities, and motion confounding.
    """
    if n_channels % n_modalities:
        raise ValueError("n_channels must be divisible by n_modalities")
    rng = np.random.default_rng(seed)
    total = n_subjects * windows_per_subject
    seq = np.zeros((total, sequence_length, n_channels), dtype=np.float32)
    labels = np.zeros(total, dtype=np.int64)
    masks = np.ones((total, n_modalities), dtype=np.float32)
    quality = np.ones((total, n_modalities), dtype=np.float32)
    timing = np.zeros((total, n_modalities), dtype=np.float32)
    motion = np.zeros(total, dtype=np.float32)
    meta: list[dict[str, object]] = []
    channel_modalities = np.repeat(np.arange(n_modalities), n_channels // n_modalities).astype(
        np.int64
    )
    idx = 0
    for s in range(n_subjects):
        subject_offset = rng.normal(0, 0.55, size=n_channels)
        class_cycle = _subject_state_sequence(rng, windows_per_subject)
        for w in range(windows_per_subject):
            y = int(class_cycle[w])
            t = np.linspace(0, 1, sequence_length, dtype=np.float32)
            base = rng.normal(0, 0.16, size=(sequence_length, n_channels)).astype(np.float32)
            class_effect = np.zeros((sequence_length, n_channels), dtype=np.float32)
            # stress proxy: BVP/EDA / temperature drift; amusement: moderate oscillatory response
            if y == 1:
                class_effect[:, 0:2] += 0.95 + 0.25 * t[:, None]
                class_effect[:, 2:4] += 0.60 + 0.35 * t[:, None]
                class_effect[:, 4:6] += 0.25
            elif y == 2:
                class_effect[:, 0:2] += 0.35 + 0.18 * np.sin(2 * np.pi * t)[:, None]
                class_effect[:, 2:4] += 0.12
                class_effect[:, 6:8] += 0.35 * np.sin(4 * np.pi * t)[:, None]
            m = float(np.clip(rng.beta(1.8, 5.0), 0, 1))
            motion[idx] = m
            # high motion creates confounding acceleration and noisy EDA/BVP, especially in non-stress windows
            class_effect[:, 6:8] += m * (1.25 + 0.15 * rng.normal(size=(sequence_length, 2)))
            if y != 1:
                base[:, 0:4] += m * rng.normal(0, 0.28, size=(sequence_length, 4))
            x = base + class_effect + subject_offset
            q = np.clip(1.0 - 0.55 * m + rng.normal(0, 0.06, n_modalities), 0.05, 1.0)
            mask = np.ones(n_modalities, dtype=np.float32)
            if rng.random() < 0.18:
                missing_modality = int(rng.integers(0, n_modalities))
                mask[missing_modality] = 0.0
                q[missing_modality] = 0.0
                x[:, channel_modalities == missing_modality] = 0.0
            lag = rng.normal(0, 250, n_modalities).astype(np.float32)
            quality[idx] = q
            masks[idx] = mask
            timing[idx] = lag
            seq[idx] = x
            labels[idx] = y
            transition = int(w > 0 and class_cycle[w - 1] != y)
            meta.append(
                {
                    "dataset_id": "synthetic_fixture",
                    "subject_id": f"S{s + 1:02d}",
                    "session_id": f"S{s + 1:02d}_session",
                    "sensor_config": "wrist",
                    "feature_representation": "synthetic",
                    "feature_schema_id": "synthetic_fixture_v1",
                    "window_end_s": float((w + 1) * 5),
                    "state_label": y,
                    "transition_label": transition,
                    "split_role": "unassigned",
                }
            )
            idx += 1
    batch = CanonicalBatch(
        sequence=seq,
        labels=labels,
        metadata=pd.DataFrame(meta),
        modality_mask=masks,
        quality=quality,
        timing_offset_ms=timing,
        motion_artifact_score=motion,
        channel_modalities=channel_modalities,
    )
    batch.validate()
    return batch
