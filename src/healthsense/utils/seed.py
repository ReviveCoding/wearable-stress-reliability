from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch with an explicit determinism contract.

    Determinism is scoped to a fixed software/hardware stack. PyTorch does not promise bitwise
    equality across releases or platforms, so run manifests retain environment fingerprints.
    """
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        # CUDA deterministic linear algebra checks this variable when CUDA is present. Setting it
        # before deterministic algorithms are enabled makes failures explicit instead of silent.
        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
        else:
            torch.use_deterministic_algorithms(False)
    except ImportError:
        # Torch is an installation dependency, but keeping this fallback makes static tooling and
        # minimal documentation environments fail only when a torch-backed model is requested.
        return
