from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from healthsense.datasets.contracts import CanonicalBatch


def load_canonical(npz_path: str | Path, metadata_path: str | Path) -> CanonicalBatch:
    """Load a canonical array bundle without allowing object-pickle payloads."""
    source = Path(npz_path)
    metadata = Path(metadata_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if not metadata.is_file():
        raise FileNotFoundError(metadata)
    with np.load(source, allow_pickle=False) as arr:
        required = {
            "sequence",
            "labels",
            "modality_mask",
            "quality",
            "timing_offset_ms",
            "motion_artifact_score",
            "channel_modalities",
        }
        missing = sorted(required - set(arr.files))
        if missing:
            raise ValueError(f"canonical NPZ missing arrays: {missing}")
        batch = CanonicalBatch(
            sequence=arr["sequence"],
            labels=arr["labels"],
            metadata=pd.read_csv(metadata),
            modality_mask=arr["modality_mask"],
            quality=arr["quality"],
            timing_offset_ms=arr["timing_offset_ms"],
            motion_artifact_score=arr["motion_artifact_score"],
            channel_modalities=arr["channel_modalities"],
        )
    batch.validate()
    return batch
