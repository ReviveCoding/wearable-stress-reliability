from __future__ import annotations

from pathlib import Path

WORKFLOW = Path(".github/workflows/ci.yml")

BANNED_WORKFLOW_FRAGMENTS = [
    "prepare-wesad",
    "wesad-development-gpu",
    "wesad-crosssubject",
    "wesad_wrist_native",
    ".local-run",
    "Project_Data",
    "C:\\Users\\",
    "Downloads\\",
    "nvidia-smi",
    "torch.cuda.is_available()",
]

REQUIRED_ENV_FLAGS = [
    "HEALTHSENSE_CI",
    "HEALTHSENSE_SKIP_FULL_WESAD",
    "HEALTHSENSE_SKIP_GPU",
]


def main() -> int:
    if not WORKFLOW.exists():
        raise SystemExit(f"Missing workflow: {WORKFLOW}")

    text = WORKFLOW.read_text(encoding="utf-8")

    violations = [
        fragment for fragment in BANNED_WORKFLOW_FRAGMENTS if fragment.lower() in text.lower()
    ]

    missing_flags = [flag for flag in REQUIRED_ENV_FLAGS if flag not in text]

    if violations:
        joined = ", ".join(violations)
        raise SystemExit(
            "CI scope violation: workflow must not run local/GPU/full-WESAD evidence steps. "
            f"Found: {joined}"
        )

    if missing_flags:
        joined = ", ".join(missing_flags)
        raise SystemExit(f"CI scope guard missing required env flags: {joined}")

    print("CI_SCOPE_GUARD=PASS")
    print("CI_SCOPE=repo-integrity-only")
    print("FULL_WESAD_IN_CI=NO")
    print("GPU_IN_CI=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
