from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from healthsense.anchors.bank import AnchorBank
from healthsense.models.classical import MajorityNoAlert, QualityAwareGBDT
from healthsense.models.neural import MaskAwareTemporalEncoder, _EncoderNet
from healthsense.models.rcpp import RCPPFormer
from healthsense.utils.io import write_json


def _write(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def _atomic_binary_write(path: Path, writer: Any) -> None:
    """Write a binary model artifact via a sibling temporary path then atomically replace it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp-{os.getpid()}{path.suffix}")
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _save_encoder(model: MaskAwareTemporalEncoder, root: Path) -> dict[str, Any]:
    if (
        model.net_ is None
        or model.n_classes_ is None
        or model.n_channels_ is None
        or model.n_modalities_ is None
    ):
        raise RuntimeError("encoder must be fitted before serialization")
    import torch

    root.mkdir(parents=True, exist_ok=True)
    spec = {
        "kind": "MaskAwareTemporalEncoder",
        "candidate_id": model.candidate_id,
        "epochs": model.epochs,
        "batch_size": model.batch_size,
        "learning_rate": model.learning_rate,
        "hidden_dim": model.hidden_dim,
        "dropout": model.dropout,
        "seed": model.seed,
        "device": model.device,
        "transition_loss_weight": model.transition_loss_weight,
        "quality_loss_weight": model.quality_loss_weight,
        "n_classes": model.n_classes_,
        "n_channels": model.n_channels_,
        "n_modalities": model.n_modalities_,
    }
    _atomic_binary_write(
        root / "state_dict.pt", lambda path: torch.save(model.net_.module.state_dict(), path)
    )
    _write(root / "model_spec.json", spec)
    return spec


def _load_encoder(root: Path, device: str = "cpu") -> MaskAwareTemporalEncoder:
    import torch

    spec = json.loads((root / "model_spec.json").read_text(encoding="utf-8"))
    model = MaskAwareTemporalEncoder(
        candidate_id=spec["candidate_id"],
        epochs=int(spec["epochs"]),
        batch_size=int(spec["batch_size"]),
        learning_rate=float(spec["learning_rate"]),
        hidden_dim=int(spec["hidden_dim"]),
        dropout=float(spec["dropout"]),
        seed=int(spec["seed"]),
        device=device,
        transition_loss_weight=float(spec["transition_loss_weight"]),
        quality_loss_weight=float(spec["quality_loss_weight"]),
    )
    model.n_classes_ = int(spec["n_classes"])
    model.n_channels_ = int(spec["n_channels"])
    model.n_modalities_ = int(spec["n_modalities"])
    model.net_ = _EncoderNet(
        model.n_channels_, model.n_modalities_, model.hidden_dim, model.n_classes_, model.dropout
    )
    model.net_.module.load_state_dict(
        torch.load(root / "state_dict.pt", map_location=device, weights_only=True)
    )
    model.net_.module.eval()
    return model


def save_candidate_model(model: Any, root: str | Path) -> Path:
    """Serialize a trained candidate in a documented, reloadable format."""
    out = Path(root)
    out.mkdir(parents=True, exist_ok=True)
    if isinstance(model, MajorityNoAlert):
        _write(
            out / "model.json",
            {
                "kind": "MajorityNoAlert",
                "candidate_id": model.candidate_id,
                "majority_class": model.majority_class_,
                "n_classes": model.n_classes_,
            },
        )
    elif isinstance(model, QualityAwareGBDT):
        if model.model_ is None:
            raise RuntimeError("C1 must be fitted before serialization")
        _atomic_binary_write(out / "model.joblib", lambda path: joblib.dump(model.model_, path))
        _write(
            out / "model.json",
            {
                "kind": "QualityAwareGBDT",
                "candidate_id": model.candidate_id,
                "max_iter": model.max_iter,
                "random_state": model.random_state,
                "n_classes": model.n_classes_,
            },
        )
    elif isinstance(model, MaskAwareTemporalEncoder):
        _save_encoder(model, out)
    elif isinstance(model, RCPPFormer):
        if (
            model.encoder_ is None
            or model.anchor_bank_ is None
            or model.classifier_ is None
            or model.n_classes_ is None
        ):
            raise RuntimeError("C3 must be fitted before serialization")
        _save_encoder(model.encoder_, out / "encoder")
        _atomic_binary_write(
            out / "anchor_bank.npz",
            lambda path: np.savez_compressed(
                path,
                anchors=model.anchor_bank_.anchors,
                anchor_labels=model.anchor_bank_.anchor_labels,
                anchor_quality=model.anchor_bank_.anchor_quality,
                source_indices=model.anchor_bank_.source_indices,
            ),
        )
        _atomic_binary_write(
            out / "classifier.joblib", lambda path: joblib.dump(model.classifier_, path)
        )
        _write(
            out / "model.json",
            {
                "kind": "RCPPFormer",
                "candidate_id": model.candidate_id,
                "epochs": model.epochs,
                "batch_size": model.batch_size,
                "learning_rate": model.learning_rate,
                "hidden_dim": model.hidden_dim,
                "dropout": model.dropout,
                "seed": model.seed,
                "device": model.device,
                "anchor_method": model.anchor_method,
                "anchors_per_class": model.anchors_per_class,
                "n_classes": model.n_classes_,
            },
        )
    else:
        raise TypeError(f"unsupported model type: {type(model).__name__}")
    return out


def load_candidate_model(root: str | Path, *, device: str = "cpu") -> Any:
    source = Path(root)
    spec_path = source / "model.json"
    if not spec_path.exists():
        spec_path = source / "model_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    kind = spec["kind"]
    if kind == "MajorityNoAlert":
        return MajorityNoAlert(
            candidate_id=spec["candidate_id"],
            majority_class_=int(spec["majority_class"]),
            n_classes_=int(spec["n_classes"]),
        )
    if kind == "QualityAwareGBDT":
        model = QualityAwareGBDT(
            candidate_id=spec["candidate_id"],
            max_iter=int(spec["max_iter"]),
            random_state=int(spec["random_state"]),
        )
        model.n_classes_ = int(spec["n_classes"])
        model.model_ = joblib.load(source / "model.joblib")
        return model
    if kind == "MaskAwareTemporalEncoder":
        return _load_encoder(source, device=device)
    if kind == "RCPPFormer":
        model = RCPPFormer(
            candidate_id=spec["candidate_id"],
            epochs=int(spec["epochs"]),
            batch_size=int(spec["batch_size"]),
            learning_rate=float(spec["learning_rate"]),
            hidden_dim=int(spec["hidden_dim"]),
            dropout=float(spec["dropout"]),
            seed=int(spec["seed"]),
            device=device,
            anchor_method=spec["anchor_method"],
            anchors_per_class=int(spec["anchors_per_class"]),
        )
        model.n_classes_ = int(spec["n_classes"])
        model.encoder_ = _load_encoder(source / "encoder", device=device)
        with np.load(source / "anchor_bank.npz", allow_pickle=False) as loaded:
            required = {"anchors", "anchor_labels", "anchor_quality", "source_indices"}
            missing = sorted(required - set(loaded.files))
            if missing:
                raise ValueError(
                    "serialized RCPP anchor bank is incomplete or predates the self-exclusion contract: "
                    f"missing={missing}"
                )
            model.anchor_bank_ = AnchorBank(
                anchors=loaded["anchors"],
                anchor_labels=loaded["anchor_labels"],
                anchor_quality=loaded["anchor_quality"],
                selection_method=model.anchor_method,
                source_indices=loaded["source_indices"],
            )
        model.classifier_ = joblib.load(source / "classifier.joblib")
        return model
    raise ValueError(f"unsupported serialized model kind: {kind}")
