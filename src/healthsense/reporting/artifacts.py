from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from healthsense.utils.io import ensure_dir, sha256_file, write_json, write_table


def _safe_run_id(run_id: str) -> str:
    candidate = Path(run_id)
    if (
        not run_id
        or candidate.is_absolute()
        or len(candidate.parts) != 1
        or candidate.name in {".", ".."}
    ):
        raise ValueError("run_id must be a non-empty single directory name without path traversal")
    return candidate.name


@dataclass
class ArtifactWriter:
    root: Path
    files: list[Path] = field(default_factory=list)

    @classmethod
    def create(
        cls, output_root: str | Path, run_id: str, *, overwrite: bool = False
    ) -> "ArtifactWriter":
        safe_run_id = _safe_run_id(run_id)
        output_base = Path(output_root).resolve()
        run_path = output_base / safe_run_id
        # Never follow a pre-existing run-id symlink. --overwrite must not become an arbitrary
        # directory deletion primitive when an output directory is attacker-controlled or stale.
        if run_path.is_symlink():
            raise ValueError("refusing a symlinked run directory")
        root = run_path.resolve()
        if root.parent != output_base:
            raise ValueError("run directory must remain directly under output_root")
        if root.exists() and any(root.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    f"refusing to reuse non-empty run directory: {root}. Use a new run ID or --overwrite "
                    "to replace the entire prior artifact set."
                )
            shutil.rmtree(root)
        return cls(root=ensure_dir(root))

    def path(self, relative: str) -> Path:
        relative_path = Path(relative)
        if not relative or relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("artifact relative path escapes the run directory")
        root = self.root.resolve()
        candidate = (root / relative_path).resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError("artifact relative path escapes the run directory")
        return candidate

    def json(self, relative: str, payload: Any) -> Path:
        out = write_json(self.path(relative), payload)
        self.files.append(out)
        return out

    def table(self, relative: str, df: pd.DataFrame) -> Path:
        out = write_table(df, self.path(relative))
        self.files.append(out)
        return out

    def text(self, relative: str, content: str) -> Path:
        out = self.path(relative)
        # Reuse atomic JSON writer's sibling-safe text path by writing a plain text payload.
        from healthsense.utils.io import _atomic_replace_text

        _atomic_replace_text(out, content)
        self.files.append(out)
        return out

    def track_tree(self, relative: str) -> None:
        root = self.path(relative)
        if not root.exists():
            raise FileNotFoundError(root)
        for file in sorted(root.rglob("*")):
            if file.is_file():
                self.files.append(file)

    def manifest(self, relative: str = "artifact_manifest.json") -> Path:
        entries = []
        for file in sorted(set(self.files)):
            if file.exists() and file.is_file():
                entries.append(
                    {
                        "relative_path": str(file.resolve().relative_to(self.root.resolve())),
                        "size_bytes": file.stat().st_size,
                        "sha256": sha256_file(file),
                    }
                )
        return write_json(self.path(relative), {"artifact_count": len(entries), "files": entries})
