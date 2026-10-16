from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from healthsense.anchors.bank import AnchorBank, build_anchor_bank
from healthsense.datasets.contracts import CanonicalBatch
from healthsense.models.base import PredictionBundle
from healthsense.models.neural import MaskAwareTemporalEncoder


@dataclass
class RCPPFormer:
    """PHT-derived Reference-Conditioned Physiological Phase residual model.

    The learned portion is a train-only residual representation and classifier. The returned
    phase distribution is a reversible, evidence-linked phase view constructed from state,
    transition, anchor-support, and data-quality signals. It is not a clinical state-space model
    and must not be described as a fitted physiological diagnosis model.
    """

    candidate_id: str = "C3"
    epochs: int = 8
    batch_size: int = 32
    learning_rate: float = 3e-3
    hidden_dim: int = 64
    dropout: float = 0.1
    seed: int = 17
    device: str = "cpu"
    anchor_method: str = "reliability_weighted_medoids"
    anchors_per_class: int = 2
    encoder_: MaskAwareTemporalEncoder | None = None
    anchor_bank_: AnchorBank | None = None
    classifier_: LogisticRegression | None = None
    n_classes_: int | None = None
    fit_seconds_: float | None = None

    @staticmethod
    def _reliability(batch: CanonicalBatch) -> np.ndarray:
        return np.clip(
            batch.quality.mean(axis=1)
            * batch.modality_mask.mean(axis=1)
            * (1.0 - 0.45 * batch.motion_artifact_score),
            0.05,
            1.0,
        ).astype(np.float32)

    def _features(
        self,
        batch: CanonicalBatch,
        bundle: PredictionBundle,
        *,
        source_indices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.anchor_bank_ is None or bundle.embeddings is None:
            raise RuntimeError("anchor bank and encoder embeddings are required")
        expected, anchor_support = self.anchor_bank_.expected(
            bundle.embeddings, exclude_source_indices=source_indices
        )
        residual = bundle.embeddings - expected
        transition = (
            bundle.phase_probabilities
            if bundle.phase_probabilities is not None
            else np.zeros((batch.n_samples, 2), dtype=np.float32)
        )
        support = (
            bundle.support_score[:, None]
            if bundle.support_score is not None
            else np.ones((batch.n_samples, 1), dtype=np.float32)
        )
        summary_quality = np.concatenate(
            [
                batch.quality.mean(axis=1, keepdims=True),
                batch.modality_mask.mean(axis=1, keepdims=True),
                batch.motion_artifact_score[:, None],
                self._reliability(batch)[:, None],
                support,
                anchor_support[:, None],
            ],
            axis=1,
        )
        x = np.concatenate(
            [bundle.embeddings, expected, residual, transition, summary_quality], axis=1
        )
        return x.astype(np.float32), expected.astype(np.float32), anchor_support.astype(np.float32)

    def fit(self, batch: CanonicalBatch) -> RCPPFormer:
        batch.validate()
        if batch.has_unknown_labels:
            raise ValueError("RCPPFormer requires reviewed non-negative labels")
        start = time.perf_counter()
        self.n_classes_ = batch.n_classes
        self.encoder_ = MaskAwareTemporalEncoder(
            candidate_id="C3_encoder",
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            seed=self.seed,
            device=self.device,
        ).fit(batch)
        train_bundle = self.encoder_.predict(batch)
        if train_bundle.embeddings is None:
            raise RuntimeError("encoder did not produce embeddings")
        self.anchor_bank_ = build_anchor_bank(
            train_bundle.embeddings,
            batch.labels,
            self._reliability(batch),
            anchors_per_class=self.anchors_per_class,
            method=self.anchor_method,
            seed=self.seed,
        )
        # Exclude an anchor from its own train-row expectation. This prevents direct self-anchor
        # support from making the residual classifier look better on the training representation.
        x, _, _ = self._features(
            batch, train_bundle, source_indices=np.arange(batch.n_samples, dtype=np.int64)
        )
        self.classifier_ = LogisticRegression(
            C=0.6,
            max_iter=500,
            class_weight="balanced",
            random_state=self.seed,
        ).fit(x, batch.labels)
        self.fit_seconds_ = time.perf_counter() - start
        return self

    def predict(self, batch: CanonicalBatch) -> PredictionBundle:
        if (
            self.encoder_ is None
            or self.anchor_bank_ is None
            or self.classifier_ is None
            or self.n_classes_ is None
        ):
            raise RuntimeError("fit before predict")
        base = self.encoder_.predict(batch)
        x, _, anchor_support = self._features(batch, base)
        probs = self.classifier_.predict_proba(x).astype(np.float32)
        pred = probs.argmax(axis=1)
        transition_up = (
            base.phase_probabilities[:, 1]
            if base.phase_probabilities is not None and base.phase_probabilities.shape[1] > 1
            else np.zeros(len(probs), dtype=np.float32)
        )
        elevated = probs[:, 1] if probs.shape[1] > 1 else np.zeros(len(probs), dtype=np.float32)
        baseline = probs[:, 0]
        evidence = self._reliability(batch)
        # Reversible phase proxy: baseline, onset, elevated, recovery, uncertain/artifact.
        phase = np.column_stack(
            [
                baseline * (1.0 - transition_up),
                elevated * transition_up,
                elevated * (1.0 - transition_up),
                baseline * transition_up,
                1.0 - np.minimum(anchor_support, evidence),
            ]
        )
        phase = np.clip(phase, 1e-6, None)
        phase = phase / phase.sum(axis=1, keepdims=True)
        support = np.minimum(anchor_support, evidence).astype(np.float32)
        rationale = [
            f"anchor_method={self.anchor_bank_.selection_method}; anchor_support={a:.3f}; evidence={e:.3f}"
            for a, e in zip(anchor_support, evidence)
        ]
        out = PredictionBundle(
            candidate_id=self.candidate_id,
            probabilities=probs,
            predicted_labels=pred,
            embeddings=base.embeddings,
            phase_probabilities=phase.astype(np.float32),
            support_score=support,
            rationale=rationale,
        )
        out.validate(batch.n_samples, self.n_classes_)
        return out
