"""Stable trace grouping shared by temporal correction and evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ordered_trace_indices(groups: np.ndarray, ordering: np.ndarray) -> list[np.ndarray]:
    """Group normalized trace IDs in first-seen order and sort times stably."""
    return [
        positions[np.argsort(ordering[positions], kind="stable")]
        for positions in pd.Series(groups).groupby(groups, sort=False).indices.values()
    ]
