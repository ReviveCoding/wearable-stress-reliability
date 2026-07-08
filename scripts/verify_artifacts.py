from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_file(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if not relative or candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("relative_path must be a non-empty relative path without traversal")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    if resolved_root != resolved and resolved_root not in resolved.parents:
        raise ValueError("relative_path resolves outside the artifact root")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a HealthSense-RCP artifact checksum manifest."
    )
    parser.add_argument("run_dir", help="Output directory containing artifact_manifest.json")
    args = parser.parse_args()
    root = Path(args.run_dir)
    manifest_path = root / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(
            json.dumps({"status": "FAIL", "reason": f"missing manifest: {manifest_path}"}, indent=2)
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(json.dumps({"status": "FAIL", "reason": f"invalid JSON: {exc}"}, indent=2))
    rows = manifest.get("files")
    expected_count = manifest.get("artifact_count")
    if (
        not isinstance(rows, list)
        or not isinstance(expected_count, int)
        or expected_count < 0
        or expected_count != len(rows)
    ):
        raise SystemExit(
            json.dumps({"status": "FAIL", "reason": "invalid artifact manifest schema"}, indent=2)
        )

    missing: list[str] = []
    checksum_mismatch: list[str] = []
    size_mismatch: list[str] = []
    invalid_rows: list[str] = []
    seen_paths: set[str] = set()
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            invalid_rows.append(f"row {row_index}: not an object")
            continue
        relative = row.get("relative_path")
        size = row.get("size_bytes")
        checksum = row.get("sha256")
        if not isinstance(relative, str):
            invalid_rows.append(f"row {row_index}: invalid relative_path")
            continue
        if relative in seen_paths:
            invalid_rows.append(f"row {row_index}: duplicate relative_path {relative!r}")
            continue
        seen_paths.add(relative)
        if not isinstance(size, int) or size < 0:
            invalid_rows.append(f"row {row_index}: invalid size_bytes for {relative!r}")
            continue
        if (
            not isinstance(checksum, str)
            or len(checksum) != 64
            or any(char not in "0123456789abcdef" for char in checksum.lower())
        ):
            invalid_rows.append(f"row {row_index}: invalid sha256 for {relative!r}")
            continue
        try:
            file = _manifest_file(root, relative)
        except ValueError as exc:
            invalid_rows.append(f"row {row_index}: {exc}")
            continue
        if not file.is_file():
            missing.append(relative)
            continue
        if size != file.stat().st_size:
            size_mismatch.append(relative)
        if checksum != _sha256(file):
            checksum_mismatch.append(relative)
    if invalid_rows or missing or checksum_mismatch or size_mismatch:
        raise SystemExit(
            json.dumps(
                {
                    "status": "FAIL",
                    "invalid_rows": invalid_rows,
                    "missing": missing,
                    "checksum_mismatch": checksum_mismatch,
                    "size_mismatch": size_mismatch,
                },
                indent=2,
            )
        )
    print(
        json.dumps(
            {"status": "PASS", "artifact_count": expected_count, "checksums_verified": True},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
