from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from healthsense.models.base import PredictionBundle
from healthsense.temporal import ordered_trace_indices


@dataclass
class TemporalAsymmetricBoundaryCorrection:
    """ABC-inspired temporal postprocessor with strict trace-boundary isolation.

    It can fill a one-window temporal gap when adjacent evidence supports coverage, then remove a
    low-support singleton that creates temporal overreach. It is a post-processing ablation, not
    part of the core RCPP claim.
    """

    candidate_id: str = "C4"
    min_support: float = 0.35
    max_extension_windows: int = 1

    @staticmethod
    def _correct_trace(
        raw: np.ndarray, support: np.ndarray, quality: np.ndarray, max_gap: int, min_support: float
    ) -> np.ndarray:
        corrected = raw.copy()
        n = len(raw)
        raw_active = raw.astype(bool, copy=False)
        # Only interior gaps supported by both original neighbors may expand.
        if n >= 3 and max_gap >= 1:
            fill_gap = (
                ~raw_active[1:-1]
                & raw_active[:-2]
                & raw_active[2:]
                & (support[1:-1] >= min_support)
                & (quality[1:-1] >= min_support)
            )
            corrected[1:-1][fill_gap] = True
        # Removing an isolated positive cannot change another positive's neighbors.
        has_neighbor = np.zeros(n, dtype=bool)
        active = corrected.astype(bool, copy=False)
        has_neighbor[1:] |= active[:-1]
        has_neighbor[:-1] |= active[1:]
        low_support = (support < min_support) | (quality < min_support)
        corrected[active & ~has_neighbor & low_support] = False
        return corrected

    def correct(
        self,
        bundle: PredictionBundle,
        quality: np.ndarray,
        *,
        elevated_class: int = 1,
        group_ids: np.ndarray | None = None,
        time_order: np.ndarray | None = None,
    ) -> PredictionBundle:
        n = len(bundle.predicted_labels)
        if quality.shape != (n,):
            raise ValueError("quality must have shape [N]")
        if group_ids is None:
            group_ids = np.zeros(n, dtype=int)
        if time_order is None:
            time_order = np.arange(n, dtype=float)
        groups = pd.Series(group_ids).astype(str).to_numpy()
        ordering = np.asarray(time_order, dtype=float)
        if ordering.shape != (n,) or np.any(~np.isfinite(ordering)):
            raise ValueError("time_order must be finite with shape [N]")
        support = (
            np.ones(n, dtype=np.float32)
            if bundle.support_score is None
            else bundle.support_score.astype(np.float32)
        )
        probs = bundle.probabilities.copy().astype(np.float32)
        stress = probs[:, elevated_class].copy()
        corrected = np.zeros(n, dtype=bool)
        raw = stress >= 0.5
        for idx in ordered_trace_indices(groups, ordering):
            corrected[idx] = self._correct_trace(
                raw[idx], support[idx], quality[idx], self.max_extension_windows, self.min_support
            )
        target_stress = np.where(
            corrected, np.maximum(stress, 0.5), np.minimum(stress, 0.499)
        ).astype(np.float32)
        nonstress = np.arange(probs.shape[1]) != elevated_class
        remainder = probs[:, nonstress]
        remainder_sum = remainder.sum(axis=1, keepdims=True)
        fallback = np.full_like(remainder, 1.0 / max(remainder.shape[1], 1))
        remainder = np.where(
            remainder_sum > 1e-8, remainder / np.clip(remainder_sum, 1e-8, None), fallback
        )
        probs[:, elevated_class] = target_stress
        probs[:, nonstress] = remainder * (1.0 - target_stress)[:, None]
        probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-8, None)
        pred = probs.argmax(axis=1)
        rationale = [
            f"T-ABC support={s:.3f}; quality={q:.3f}; corrected={int(c)}"
            for s, q, c in zip(support, quality, corrected)
        ]
        out = PredictionBundle(
            candidate_id=self.candidate_id,
            probabilities=probs.astype(np.float32),
            predicted_labels=pred.astype(np.int64),
            embeddings=bundle.embeddings,
            phase_probabilities=bundle.phase_probabilities,
            support_score=support.astype(np.float32),
            rationale=rationale,
        )
        out.validate(n, probs.shape[1])
        return out
