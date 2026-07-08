from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from healthsense.reporting.artifacts import ArtifactWriter


def test_artifact_writer_rejects_run_and_artifact_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run_id"):
        ArtifactWriter.create(tmp_path, "../escape")
    writer = ArtifactWriter.create(tmp_path, "safe")
    with pytest.raises(ValueError, match="escapes"):
        writer.path("../escape.txt")


def test_artifact_writer_requires_explicit_overwrite(tmp_path: Path) -> None:
    writer = ArtifactWriter.create(tmp_path, "run")
    writer.table("result.csv", pd.DataFrame({"x": [1]}))
    writer.manifest()
    with pytest.raises(FileExistsError, match="--overwrite"):
        ArtifactWriter.create(tmp_path, "run")
    fresh = ArtifactWriter.create(tmp_path, "run", overwrite=True)
    assert not (fresh.root / "result.csv").exists()


def test_checksum_verifier_rejects_traversal_and_duplicate_entries(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "safe.txt").write_text("ok", encoding="utf-8")
    manifest = {
        "artifact_count": 2,
        "files": [
            {"relative_path": "../outside.txt", "size_bytes": 0, "sha256": "0" * 64},
            {"relative_path": "../outside.txt", "size_bytes": 0, "sha256": "0" * 64},
        ],
    }
    (run / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "verify_artifacts.py"
    completed = subprocess.run(
        [sys.executable, str(script), str(run)], capture_output=True, text=True, check=False
    )
    assert completed.returncode != 0
    assert "traversal" in completed.stderr


def test_artifact_writer_rejects_symlinked_run_directory(tmp_path: Path) -> None:
    if not hasattr(__import__("os"), "symlink"):
        pytest.skip("symlink support unavailable")
    outside = tmp_path / "outside"
    outside.mkdir()
    run_link = tmp_path / "linked-run"
    try:
        run_link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable in this runtime")
    with pytest.raises(ValueError, match="symlinked"):
        ArtifactWriter.create(tmp_path, "linked-run", overwrite=True)


def test_artifact_writer_atomic_writes_leave_no_temporary_files(tmp_path: Path) -> None:
    writer = ArtifactWriter.create(tmp_path, "atomic")
    writer.text("note.txt", "hello")
    writer.json("state.json", {"ok": True})
    writer.table("table.csv", pd.DataFrame({"x": [1, 2]}))
    assert not list(writer.root.rglob(".*.tmp-*"))
