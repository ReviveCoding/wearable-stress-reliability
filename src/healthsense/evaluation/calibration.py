from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import log_loss


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@dataclass
class TemperatureScaler:
    temperature_: float = 1.0

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        clipped = np.clip(probabilities, 1e-6, 1.0)
        logits = np.log(clipped)
        grid = np.geomspace(0.25, 4.0, 81)
        losses = [
            log_loss(labels, _softmax(logits / t), labels=np.arange(probabilities.shape[1]))
            for t in grid
        ]
        self.temperature_ = float(grid[int(np.argmin(losses))])
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        logits = np.log(np.clip(probabilities, 1e-6, 1.0))
        return _softmax(logits / self.temperature_).astype(np.float32)
