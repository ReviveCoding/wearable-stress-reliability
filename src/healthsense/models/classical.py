from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from healthsense.datasets.contracts import CanonicalBatch
from healthsense.models.base import PredictionBundle


@dataclass
class MajorityNoAlert:
    candidate_id: str = "C0"
    majority_class_: int | None = None
    n_classes_: int | None = None
    fit_seconds_: float | None = None

    def fit(self, batch: CanonicalBatch) -> MajorityNoAlert:
        batch.validate()
        if batch.has_unknown_labels:
            raise ValueError("MajorityNoAlert requires reviewed non-negative labels")
        started = time.perf_counter()
        values, counts = np.unique(batch.labels, return_counts=True)
        self.majority_class_ = int(values[np.argmax(counts)])
        self.n_classes_ = batch.n_classes
        self.fit_seconds_ = float(time.perf_counter() - started)
        return self

    def predict(self, batch: CanonicalBatch) -> PredictionBundle:
        if self.majority_class_ is None or self.n_classes_ is None:
            raise RuntimeError("fit before predict")
        probs = np.full((batch.n_samples, self.n_classes_), 1e-4, dtype=np.float32)
        probs[:, self.majority_class_] = 1.0 - (self.n_classes_ - 1) * 1e-4
        pred = probs.argmax(axis=1)
        out = PredictionBundle(
            candidate_id=self.candidate_id, probabilities=probs, predicted_labels=pred
        )
        out.validate(batch.n_samples, self.n_classes_)
        return out


@dataclass
class QualityAwareGBDT:
    candidate_id: str = "C1"
    max_iter: int = 200
    random_state: int = 17
    model_: HistGradientBoostingClassifier | None = None
    n_classes_: int | None = None
    fit_seconds_: float | None = None

    def fit(self, batch: CanonicalBatch) -> QualityAwareGBDT:
        batch.validate()
        if batch.has_unknown_labels:
            raise ValueError("QualityAwareGBDT requires reviewed non-negative labels")
        started = time.perf_counter()
        x = batch.as_feature_frame().to_numpy(dtype=np.float32)
        self.n_classes_ = batch.n_classes
        self.model_ = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            learning_rate=0.08,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=self.random_state,
        )
        # Public wearable states can be uneven after participant-disjoint splitting. Balanced
        # weights keep the industry baseline competitive without altering the held-out metric.
        sample_weight = compute_sample_weight(class_weight="balanced", y=batch.labels)
        # Limit thread pools for deterministic, small-cohort CI/smoke execution.
        with threadpool_limits(limits=1):
            self.model_.fit(x, batch.labels, sample_weight=sample_weight)
        self.fit_seconds_ = float(time.perf_counter() - started)
        return self

    def predict(self, batch: CanonicalBatch) -> PredictionBundle:
        if self.model_ is None or self.n_classes_ is None:
            raise RuntimeError("fit before predict")
        x = batch.as_feature_frame().to_numpy(dtype=np.float32)
        with threadpool_limits(limits=1):
            probs = self.model_.predict_proba(x).astype(np.float32)
        pred = probs.argmax(axis=1)
        out = PredictionBundle(
            candidate_id=self.candidate_id, probabilities=probs, predicted_labels=pred
        )
        out.validate(batch.n_samples, self.n_classes_)
        return out
