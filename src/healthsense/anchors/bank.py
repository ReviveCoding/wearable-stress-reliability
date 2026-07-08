from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AnchorBank:
    anchors: np.ndarray
    anchor_labels: np.ndarray
    anchor_quality: np.ndarray
    selection_method: str
    source_indices: np.ndarray

    def expected(
        self,
        embeddings: np.ndarray,
        *,
        exclude_source_indices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a train-only reference expectation.

        ``exclude_source_indices`` is used only while constructing train-row representations.
        It removes an anchor from the expectation of the same original row, preventing direct
        self-support leakage while leaving test-time reference use unchanged.
        """
        x = _normalize(embeddings)
        a = _normalize(self.anchors)
        similarity = x @ a.T
        # Quality-weighted softmax similarity. No test labels or test anchors.
        logits = similarity - similarity.max(axis=1, keepdims=True)
        weights = np.exp(logits) * self.anchor_quality[None, :]
        if exclude_source_indices is not None:
            source = np.asarray(exclude_source_indices, dtype=np.int64)
            if source.shape != (len(embeddings),):
                raise ValueError("exclude_source_indices must have shape [N]")
            weights = np.where(source[:, None] == self.source_indices[None, :], 0.0, weights)
            # A valid bank has anchors from multiple samples/classes, but fail closed if a caller
            # constructs a degenerate one-anchor bank and excludes its only anchor.
            if np.any(weights.sum(axis=1) <= 0):
                raise ValueError("anchor exclusion removed every available reference")
        weights = weights / np.clip(weights.sum(axis=1, keepdims=True), 1e-8, None)
        expected = weights @ self.anchors
        support = weights.max(axis=1)
        return expected.astype(np.float32), support.astype(np.float32)


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def build_anchor_bank(
    embeddings: np.ndarray,
    labels: np.ndarray,
    quality: np.ndarray,
    *,
    anchors_per_class: int = 2,
    method: str = "reliability_weighted_medoids",
    seed: int = 17,
) -> AnchorBank:
    if len(embeddings) != len(labels) or len(labels) != len(quality):
        raise ValueError("anchor inputs must have matching sample dimension")
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for label in sorted(np.unique(labels).tolist()):
        idx = np.flatnonzero(labels == label)
        if len(idx) == 0:
            continue
        if method == "random":
            picks = rng.choice(idx, size=min(anchors_per_class, len(idx)), replace=False)
        elif method == "nearest":
            centroid = embeddings[idx].mean(axis=0)
            dist = np.linalg.norm(embeddings[idx] - centroid, axis=1)
            picks = idx[np.argsort(dist)[: min(anchors_per_class, len(idx))]]
        elif method in {"kmedoids", "reliability_weighted_medoids"}:
            # Small, deterministic greedy medoid approximation. Reliability weighting is train-only.
            local_quality = (
                quality[idx] if method == "reliability_weighted_medoids" else np.ones(len(idx))
            )
            pairwise = np.linalg.norm(embeddings[idx, None, :] - embeddings[None, idx, :], axis=2)
            objective = (pairwise * local_quality[None, :]).sum(axis=1)
            first = int(np.argmin(objective))
            selected = [first]
            while len(selected) < min(anchors_per_class, len(idx)):
                min_dist = pairwise[:, selected].min(axis=1)
                score = min_dist * local_quality
                selected.append(int(np.argmax(score)))
            picks = idx[np.asarray(selected, dtype=int)]
        else:
            raise ValueError(f"unsupported anchor method: {method}")
        chosen.extend(np.asarray(picks, dtype=int).tolist())
    selected = np.asarray(sorted(set(chosen)), dtype=int)
    return AnchorBank(
        anchors=embeddings[selected].astype(np.float32),
        anchor_labels=labels[selected].astype(np.int64),
        anchor_quality=np.clip(quality[selected], 0.05, 1.0).astype(np.float32),
        selection_method=method,
        source_indices=selected.astype(np.int64),
    )
