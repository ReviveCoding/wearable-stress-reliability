from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_json_hash(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, default=str, separators=(",", ":")))


def _atomic_replace_text(out: Path, content: str) -> Path:
    ensure_dir(out.parent)
    tmp = out.with_name(f".{out.name}.tmp-{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return out


def write_json(path: str | Path, payload: Any) -> Path:
    out = Path(path)
    return _atomic_replace_text(
        out, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    out = Path(path)
    ensure_dir(out.parent)
    if out.suffix == ".parquet":
        tmp = out.with_name(f".{out.stem}.tmp-{os.getpid()}{out.suffix}")
        try:
            df.to_parquet(tmp, index=False)
            os.replace(tmp, out)
            return out
        except Exception:
            tmp.unlink(missing_ok=True)
            out = out.with_suffix(".csv")
    csv = df.to_csv(index=False)
    return _atomic_replace_text(out, csv)


def file_fingerprint(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return {
        "path": str(p),
        "exists": p.exists(),
        "size_bytes": p.stat().st_size if p.exists() else None,
        "sha256": sha256_file(p) if p.exists() and p.is_file() else None,
    }


def tree_fingerprint(
    root: str | Path, patterns: tuple[str, ...] = ("*.py", "*.yaml", "*.yml")
) -> str:
    root_path = Path(root)
    pieces: list[str] = []
    for pattern in patterns:
        for path in sorted(root_path.rglob(pattern)):
            if path.is_file() and ".git" not in path.parts and "outputs" not in path.parts:
                pieces.append(f"{path.relative_to(root_path)}:{sha256_file(path)}")
    return sha256_text("\n".join(pieces))


def hardware_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "pid": os.getpid(),
    }
    try:
        import torch

        snapshot.update(
            {
                "torch": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "torch_deterministic_algorithms": bool(
                    torch.are_deterministic_algorithms_enabled()
                ),
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            }
        )
    except Exception as exc:  # pragma: no cover
        snapshot["torch_error"] = repr(exc)
    return snapshot


def git_revision(root: str | Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(root),
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() or None
    except Exception:
        return None
