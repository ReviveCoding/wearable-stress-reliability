from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from healthsense.datasets.contracts import CanonicalBatch


@dataclass
class PredictionBundle:
    candidate_id: str
    probabilities: np.ndarray
    predicted_labels: np.ndarray
    embeddings: np.ndarray | None = None
    phase_probabilities: np.ndarray | None = None
    support_score: np.ndarray | None = None
    rationale: list[str] | None = None

    def validate(self, n_samples: int, n_classes: int) -> None:
        if self.probabilities.shape != (n_samples, n_classes):
            raise ValueError("invalid probability shape")
        if self.predicted_labels.shape != (n_samples,):
            raise ValueError("invalid prediction shape")
        if not np.allclose(self.probabilities.sum(axis=1), 1.0, atol=1e-4):
            raise ValueError("probabilities must sum to 1")


class CandidateModel(Protocol):
    candidate_id: str

    def fit(self, batch: CanonicalBatch) -> CandidateModel: ...

    def predict(self, batch: CanonicalBatch) -> PredictionBundle: ...
