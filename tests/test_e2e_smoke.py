from pathlib import Path

from healthsense.pipeline import run_smoke


def test_smoke_creates_required_artifacts(tmp_path: Path):
    root = run_smoke(output_root=str(tmp_path), run_id="pytest-smoke")
    required = [
        "metrics.json",
        "candidate_comparison.csv",
        "slice_metrics.csv",
        "episode_metrics.csv",
        "robustness_metrics.csv",
        "calibration_table.csv",
        "operational_metrics.json",
        "decision_scenarios.csv",
        "run_manifest.json",
        "release_decision.json",
        "artifact_manifest.json",
        "reports/benchmark_report.md",
        "figures/f02_candidate_comparison.png",
    ]
    for name in required:
        assert (root / name).exists(), name
