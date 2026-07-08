from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from healthsense.datasets.contracts import CanonicalBatch
from healthsense.utils.io import stable_json_hash


@dataclass(frozen=True)
class ParticipantSplit:
    train_idx: np.ndarray
    calibration_idx: np.ndarray
    dev_idx: np.ndarray
    final_idx: np.ndarray
    manifest: dict[str, Any]

    def validate(self, batch: CanonicalBatch) -> None:
        batch.validate()
        if batch.has_unknown_labels:
            raise ValueError(
                "participant split requires reviewed non-negative labels; unknown windows must be "
                "resolved or excluded before supervised evaluation"
            )
        all_sets = [self.train_idx, self.calibration_idx, self.dev_idx, self.final_idx]
        union = np.concatenate(all_sets)
        if len(np.unique(union)) != len(union):
            raise ValueError("split indices overlap")
        if len(union) != batch.n_samples:
            raise ValueError("split does not cover all samples")
        subject_sets = []
        for idx in all_sets:
            subject_sets.append(set(batch.metadata.iloc[idx]["subject_id"].astype(str)))
        for i in range(len(subject_sets)):
            for j in range(i + 1, len(subject_sets)):
                if subject_sets[i] & subject_sets[j]:
                    raise ValueError("participant leakage across split roles")
        required_classes = set(np.unique(batch.labels).astype(int).tolist())
        for role, idx in zip(("train", "calibration", "dev", "final"), all_sets):
            observed = set(np.unique(batch.labels[idx]).astype(int).tolist())
            missing_classes = sorted(required_classes - observed)
            if missing_classes:
                raise ValueError(
                    f"split role '{role}' is missing required classes {missing_classes}; "
                    "participant-level class coverage is required for this supervised evaluation contract"
                )


def create_participant_split(
    batch: CanonicalBatch,
    *,
    seed: int,
    final_holdout_subjects: int = 2,
    calibration_subjects: int = 1,
) -> ParticipantSplit:
    subjects = np.array(sorted(batch.metadata["subject_id"].astype(str).unique().tolist()))
    if len(subjects) < final_holdout_subjects + calibration_subjects + 3:
        raise ValueError("need at least final_holdout + calibration + 3 subjects")
    rng = np.random.default_rng(seed)
    shuffled = subjects.copy()
    rng.shuffle(shuffled)
    final_subjects = sorted(shuffled[:final_holdout_subjects].tolist())
    calibration_subject_set = sorted(
        shuffled[final_holdout_subjects : final_holdout_subjects + calibration_subjects].tolist()
    )
    remaining = shuffled[final_holdout_subjects + calibration_subjects :]
    # preserve a development participant-disjoint test role; at least one participant.
    dev_subjects = sorted(remaining[-1:].tolist())
    train_subjects = sorted(remaining[:-1].tolist())
    if len(train_subjects) < 2:
        raise ValueError("not enough train subjects after allocating final/calibration/dev")
    values = batch.metadata["subject_id"].astype(str).to_numpy()
    roles = {
        "train": train_subjects,
        "calibration": calibration_subject_set,
        "dev": dev_subjects,
        "final": final_subjects,
    }
    indices = {role: np.flatnonzero(np.isin(values, subs)) for role, subs in roles.items()}
    manifest: dict[str, Any] = {
        "policy": "participant_disjoint_deterministic",
        "seed": seed,
        "subjects": roles,
        "n_samples": {role: int(len(idx)) for role, idx in indices.items()},
        "prohibitions": [
            "random window split",
            "test-subject preprocessing fit",
            "test calibration or threshold fitting",
            "test anchor selection",
            "future-signal features",
        ],
    }
    manifest["split_hash"] = stable_json_hash(manifest)
    result = ParticipantSplit(
        train_idx=indices["train"],
        calibration_idx=indices["calibration"],
        dev_idx=indices["dev"],
        final_idx=indices["final"],
        manifest=manifest,
    )
    result.validate(batch)
    return result


def load_participant_split(batch: CanonicalBatch, manifest: dict[str, Any]) -> ParticipantSplit:
    """Rebuild an immutable participant split from a previously published manifest."""
    if manifest.get("policy") not in {
        "participant_disjoint_deterministic",
        "participant_disjoint_cross_subject_cv",
    }:
        raise ValueError("unsupported split policy")
    roles = manifest.get("subjects")
    if not isinstance(roles, dict) or set(roles) != {"train", "calibration", "dev", "final"}:
        raise ValueError("split manifest must contain train/calibration/dev/final subject roles")
    available = set(batch.metadata["subject_id"].astype(str))
    referenced = set().union(*(set(map(str, value)) for value in roles.values()))
    if referenced != available:
        missing = sorted(referenced - available)
        unexpected = sorted(available - referenced)
        raise ValueError(
            f"split manifest subject mismatch; missing={missing}, unexpected={unexpected}"
        )
    values = batch.metadata["subject_id"].astype(str).to_numpy()
    indices = {
        role: np.flatnonzero(np.isin(values, list(map(str, subjects))))
        for role, subjects in roles.items()
    }
    rebuilt = {
        "policy": manifest["policy"],
        "seed": manifest.get("seed"),
        "subjects": {role: sorted(map(str, subjects)) for role, subjects in roles.items()},
        "n_samples": {role: int(len(idx)) for role, idx in indices.items()},
        "prohibitions": manifest.get("prohibitions", []),
    }
    if "fold_id" in manifest:
        rebuilt["fold_id"] = manifest["fold_id"]
    rebuilt["split_hash"] = stable_json_hash(rebuilt)
    expected = manifest.get("split_hash")
    if expected is not None and expected != rebuilt["split_hash"]:
        raise ValueError("split manifest hash mismatch; source dataset or split contract changed")
    result = ParticipantSplit(
        train_idx=indices["train"],
        calibration_idx=indices["calibration"],
        dev_idx=indices["dev"],
        final_idx=indices["final"],
        manifest=rebuilt,
    )
    result.validate(batch)
    return result


def create_cross_subject_manifests(
    batch: CanonicalBatch,
    *,
    seed: int,
    n_folds: int,
) -> list[dict[str, Any]]:
    """Create deterministic nested participant-disjoint confirmation folds.

    Each fold reserves exactly one final subject, one calibration subject, and one development
    subject. All other participants are training-only for that fold. The routine is designed for
    repeated subject-held-out evidence, not for a random-window CV shortcut.
    """
    subjects = np.array(sorted(batch.metadata["subject_id"].astype(str).unique().tolist()))
    if len(subjects) < 5:
        raise ValueError("cross-subject confirmation requires at least five participants")
    if n_folds < 2 or n_folds > len(subjects):
        raise ValueError("n_folds must be in [2, n_subjects]")
    rng = np.random.default_rng(seed)
    order = subjects.copy()
    rng.shuffle(order)
    manifests: list[dict[str, Any]] = []
    for fold in range(n_folds):
        final = str(order[fold])
        calibration = str(order[(fold + 1) % len(order)])
        dev = str(order[(fold + 2) % len(order)])
        train = sorted(str(value) for value in order if str(value) not in {final, calibration, dev})
        roles = {"train": train, "calibration": [calibration], "dev": [dev], "final": [final]}
        values = batch.metadata["subject_id"].astype(str).to_numpy()
        index_counts = {
            role: int(np.isin(values, members).sum()) for role, members in roles.items()
        }
        manifest: dict[str, Any] = {
            "policy": "participant_disjoint_cross_subject_cv",
            "seed": seed,
            "fold_id": fold + 1,
            "subjects": roles,
            "n_samples": index_counts,
            "prohibitions": [
                "random window split",
                "test-subject preprocessing fit",
                "test calibration or threshold fitting",
                "test anchor selection",
                "future-signal features",
                "cross-fold candidate selection from final metrics",
            ],
        }
        manifest["split_hash"] = stable_json_hash(manifest)
        manifests.append(manifest)
    return manifests
