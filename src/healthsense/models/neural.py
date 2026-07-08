from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from healthsense.datasets.contracts import CanonicalBatch
from healthsense.models.base import PredictionBundle
from healthsense.utils.seed import set_global_seed


def _torch():
    import torch
    import torch.nn as nn

    return torch, nn


def _nhead(hidden_dim: int) -> int:
    for nhead in (4, 2, 1):
        if hidden_dim % nhead == 0:
            return nhead
    return 1


class _EncoderNet:  # constructed lazily to keep optional torch import localized
    """Small mask- and quality-aware temporal Transformer encoder.

    Modality availability and data-quality proxies gate their corresponding raw channels before
    temporal feature extraction. This avoids calling the model mask-aware merely because masks
    are concatenated after an unmasked convolution.
    """

    def __init__(
        self, n_channels: int, n_modalities: int, hidden_dim: int, n_classes: int, dropout: float
    ):
        torch, nn = _torch()
        self.torch = torch
        self.nn = nn
        self.module = nn.Module()
        self.module.input_proj = nn.Sequential(
            nn.Linear(n_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=_nhead(hidden_dim),
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.module.temporal = nn.TransformerEncoder(layer, num_layers=1)
        self.module.context = nn.Sequential(
            nn.Linear(hidden_dim + 3 * n_modalities + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.module.classifier = nn.Linear(hidden_dim, n_classes)
        self.module.transition = nn.Linear(hidden_dim, 2)
        self.module.quality = nn.Linear(hidden_dim, 1)

    def _position(self, steps: int, hidden_dim: int, device):
        torch = self.torch
        position = torch.arange(steps, dtype=torch.float32, device=device)[:, None]
        scale = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float32, device=device)
            * (-math.log(10000.0) / hidden_dim)
        )
        pos = torch.zeros((steps, hidden_dim), dtype=torch.float32, device=device)
        pos[:, 0::2] = torch.sin(position * scale)
        if hidden_dim > 1:
            pos[:, 1::2] = torch.cos(position * scale[: pos[:, 1::2].shape[1]])
        return pos[None, :, :]

    def forward(self, seq, mask, quality, timing, motion, channel_modalities):
        torch = self.torch
        # [N, M] -> [N, C] map. Quality is a proxy, so it attenuates rather than deletes a
        # channel, while missingness deletes it exactly.
        per_channel_gate = mask[:, channel_modalities] * quality[:, channel_modalities]
        gated_seq = seq * per_channel_gate[:, None, :]
        temporal = self.module.input_proj(gated_seq)
        temporal = temporal + self._position(temporal.shape[1], temporal.shape[2], temporal.device)
        temporal = self.module.temporal(temporal)
        h = temporal.mean(dim=1)
        aux = torch.cat([mask, quality, timing / 1000.0, motion[:, None]], dim=1)
        embedding = self.module.context(torch.cat([h, aux], dim=1))
        return (
            self.module.classifier(embedding),
            embedding,
            self.module.transition(embedding),
            self.module.quality(embedding),
        )


@dataclass
class MaskAwareTemporalEncoder:
    candidate_id: str = "C2"
    epochs: int = 8
    batch_size: int = 32
    learning_rate: float = 3e-3
    hidden_dim: int = 64
    dropout: float = 0.1
    seed: int = 17
    device: str = "auto"
    transition_loss_weight: float = 0.20
    quality_loss_weight: float = 0.10
    net_: _EncoderNet | None = None
    n_classes_: int | None = None
    n_channels_: int | None = None
    n_modalities_: int | None = None
    fit_seconds_: float | None = None
    runtime_device_: str | None = None

    def _device(self):
        torch, _ = _torch()
        requested = str(self.device).lower()
        if requested == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif requested == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("models.device=cuda was requested but CUDA is unavailable")
            device = torch.device("cuda")
        elif requested == "cpu":
            device = torch.device("cpu")
        else:
            raise ValueError("device must be one of: cpu, cuda, auto")
        if device.type == "cpu":
            torch.set_num_threads(1)
        self.runtime_device_ = device.type
        return device

    def _tensors(self, batch: CanonicalBatch, device):
        torch, _ = _torch()
        transition = batch.metadata["transition_label"].to_numpy(dtype=np.int64)
        quality_target = np.clip(
            batch.quality.mean(axis=1) * (1.0 - 0.45 * batch.motion_artifact_score), 0.0, 1.0
        )
        return (
            torch.tensor(batch.sequence, dtype=torch.float32, device=device),
            torch.tensor(batch.modality_mask, dtype=torch.float32, device=device),
            torch.tensor(batch.quality, dtype=torch.float32, device=device),
            torch.tensor(batch.timing_offset_ms, dtype=torch.float32, device=device),
            torch.tensor(batch.motion_artifact_score, dtype=torch.float32, device=device),
            torch.tensor(batch.channel_modalities, dtype=torch.long, device=device),
            torch.tensor(batch.labels, dtype=torch.long, device=device),
            torch.tensor(transition, dtype=torch.long, device=device),
            torch.tensor(quality_target, dtype=torch.float32, device=device),
        )

    def fit(self, batch: CanonicalBatch) -> "MaskAwareTemporalEncoder":
        batch.validate()
        if batch.has_unknown_labels:
            raise ValueError("MaskAwareTemporalEncoder requires reviewed non-negative labels")
        set_global_seed(self.seed, deterministic=True)
        torch, nn = _torch()
        device = self._device()
        self.n_classes_ = batch.n_classes
        self.n_channels_ = int(batch.sequence.shape[2])
        self.n_modalities_ = int(batch.modality_mask.shape[1])
        self.net_ = _EncoderNet(
            n_channels=self.n_channels_,
            n_modalities=self.n_modalities_,
            hidden_dim=self.hidden_dim,
            n_classes=batch.n_classes,
            dropout=self.dropout,
        )
        net = self.net_.module.to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        state_loss = nn.CrossEntropyLoss()
        transition_loss = nn.CrossEntropyLoss()
        quality_loss = nn.MSELoss()
        (
            seq_all,
            mask_all,
            quality_all,
            timing_all,
            motion_all,
            channel_modalities,
            y_all,
            transition_all,
            quality_target_all,
        ) = self._tensors(batch, device)
        n = batch.n_samples
        start = time.perf_counter()
        net.train()
        for _ in range(self.epochs):
            order = torch.randperm(n, device=device)
            for idx in order.split(self.batch_size):
                seq = seq_all[idx]
                mask = mask_all[idx]
                quality = quality_all[idx]
                timing = timing_all[idx]
                motion = motion_all[idx]
                y = y_all[idx]
                transition_y = transition_all[idx]
                quality_y = quality_target_all[idx]
                logits, _, transition_logits, support_logits = self.net_.forward(
                    seq, mask, quality, timing, motion, channel_modalities
                )
                loss = state_loss(logits, y)
                loss = loss + self.transition_loss_weight * transition_loss(
                    transition_logits, transition_y
                )
                loss = loss + self.quality_loss_weight * quality_loss(
                    torch.sigmoid(support_logits).squeeze(-1), quality_y
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
        self.fit_seconds_ = time.perf_counter() - start
        return self

    def predict(self, batch: CanonicalBatch) -> PredictionBundle:
        if self.net_ is None or self.n_classes_ is None:
            raise RuntimeError("fit before predict")
        torch, _ = _torch()
        device = self._device()
        self.net_.module.to(device).eval()
        seq, mask, quality, timing, motion, channel_modalities, _, _, _ = self._tensors(
            batch, device
        )
        with torch.inference_mode():
            logits, emb, transition, support_logits = self.net_.forward(
                seq, mask, quality, timing, motion, channel_modalities
            )
            probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
            embeddings = emb.cpu().numpy().astype(np.float32)
            support = torch.sigmoid(support_logits).squeeze(-1).cpu().numpy().astype(np.float32)
            phase = torch.softmax(transition, dim=1).cpu().numpy().astype(np.float32)
        pred = probs.argmax(axis=1)
        out = PredictionBundle(
            candidate_id=self.candidate_id,
            probabilities=probs,
            predicted_labels=pred,
            embeddings=embeddings,
            phase_probabilities=phase,
            support_score=support,
        )
        out.validate(batch.n_samples, self.n_classes_)
        return out
