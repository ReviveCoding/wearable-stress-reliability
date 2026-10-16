from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SequenceStandardizer:
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, sequence: np.ndarray) -> SequenceStandardizer:
        if sequence.ndim != 3:
            raise ValueError("sequence must be [N,T,C]")
        self.mean_ = sequence.mean(axis=(0, 1), keepdims=True)
        self.std_ = sequence.std(axis=(0, 1), keepdims=True)
        self.std_ = np.where(self.std_ < 1e-6, 1.0, self.std_)
        return self

    def transform(self, sequence: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("fit must be called on training data before transform")
        return ((sequence - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, sequence: np.ndarray) -> np.ndarray:
        return self.fit(sequence).transform(sequence)
