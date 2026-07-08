from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from healthsense.reporting.artifacts import ArtifactWriter


def _save(writer: ArtifactWriter, name: str) -> None:
    out = writer.path(f"figures/{name}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    writer.files.append(out)


def generate_figures(
    writer: ArtifactWriter,
    *,
    comparison: pd.DataFrame,
    selections: pd.DataFrame,
    robustness: pd.DataFrame,
    frontier: pd.DataFrame,
    slice_df: pd.DataFrame,
    metadata: pd.DataFrame,
    operational: pd.DataFrame,
    decision_scenarios: pd.DataFrame,
    dataset_label: str,
) -> None:
    # F01: target distribution
    plt.figure(figsize=(7, 4))
    metadata["state_label"].value_counts().sort_index().plot(kind="bar")
    plt.title(f"F01. Target distribution ({dataset_label})")
    plt.xlabel("Protocol-state class")
    plt.ylabel("Window count")
    _save(writer, "f01_target_distribution")

    # F02: final candidates. Operating thresholds were selected on dev, not this final data.
    plt.figure(figsize=(8, 4))
    plt.bar(comparison["candidate_id"], comparison["macro_f1"])
    plt.title("F02. Final macro-F1, participant-disjoint holdout")
    plt.xlabel("Candidate")
    plt.ylabel("Macro-F1")
    _save(writer, "f02_candidate_comparison")

    # F03: calibration
    plt.figure(figsize=(8, 4))
    plt.bar(comparison["candidate_id"], comparison["adaptive_ece"])
    plt.title("F03. Final adaptive calibration error")
    plt.xlabel("Candidate")
    plt.ylabel("Adaptive ECE, lower is better")
    _save(writer, "f03_calibration")

    # F04: final diagnostic frontier, explicitly not used to select the reported threshold.
    plt.figure(figsize=(7, 5))
    for candidate, group in frontier.groupby("candidate_id"):
        plt.plot(
            group["false_alert_minutes_per_nonstress_hour"],
            group["stress_episode_recall"],
            marker="o",
            label=candidate,
        )
    plt.title("F04. Diagnostic episode coverage versus false-alert burden")
    plt.xlabel("False-alert minutes per non-stress hour")
    plt.ylabel("Stress episode recall")
    plt.legend()
    _save(writer, "f04_coverage_overreach_frontier")

    # F05: robustness
    plt.figure(figsize=(10, 4))
    for candidate, group in robustness.groupby("candidate_id"):
        plt.plot(group["scenario"], group["macro_f1"], marker="o", label=candidate)
    plt.title("F05. Robustness under modality and artifact faults")
    plt.xlabel("Scenario")
    plt.ylabel("Macro-F1")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    _save(writer, "f05_robustness")

    # F06: participant slices
    plt.figure(figsize=(9, 4))
    for candidate, group in slice_df.groupby("candidate_id"):
        plt.plot(group["slice_value"], group["macro_f1"], marker="o", label=candidate)
    plt.title("F06. Final-holdout participant slice macro-F1")
    plt.xlabel("Final holdout participant")
    plt.ylabel("Macro-F1")
    plt.legend()
    _save(writer, "f06_worst_subject")

    # F07: architecture diagram
    plt.figure(figsize=(12, 2.5))
    stages = [
        "Validate",
        "Causal windows",
        "C0-C3",
        "Calibrate",
        "C4 T-ABC",
        "C5 policy",
        "Release report",
    ]
    for i, stage in enumerate(stages):
        plt.text(i, 0.5, stage, ha="center", va="center", bbox={"boxstyle": "round", "pad": 0.5})
        if i < len(stages) - 1:
            plt.annotate(
                "", xy=(i + 0.42, 0.5), xytext=(i + 0.58, 0.5), arrowprops={"arrowstyle": "->"}
            )
    plt.xlim(-0.7, len(stages) - 0.3)
    plt.ylim(0, 1)
    plt.axis("off")
    plt.title("F07. HealthSense-RCP canonical evaluation path")
    _save(writer, "f07_architecture")

    # F08: development-only operating threshold selection
    plt.figure(figsize=(8, 4))
    plt.bar(selections["candidate_id"], selections["threshold"])
    plt.title("F08. Operating threshold selected on participant-disjoint dev cohort")
    plt.xlabel("Candidate")
    plt.ylabel("Frozen probability threshold")
    _save(writer, "f08_dev_operating_thresholds")

    # F09: final latency/frontier
    plt.figure(figsize=(8, 4))
    merged = comparison.merge(
        operational[["candidate_id", "inference_p95_ms"]], on="candidate_id", how="left"
    )
    plt.scatter(merged["inference_p95_ms"], merged["macro_f1"])
    for _, row in merged.iterrows():
        plt.annotate(str(row["candidate_id"]), (row["inference_p95_ms"], row["macro_f1"]))
    plt.title("F09. Final macro-F1 versus measured inference latency")
    plt.xlabel("Inference p95 latency, ms")
    plt.ylabel("Macro-F1")
    _save(writer, "f09_performance_latency")

    # F10: abstention scenarios
    plt.figure(figsize=(8, 4))
    plt.bar(decision_scenarios["scenario"], decision_scenarios["selective_coverage"])
    plt.title("F10. Selective-policy coverage by fixed scenario")
    plt.xlabel("Policy scenario")
    plt.ylabel("Accepted prediction coverage")
    plt.xticks(rotation=20, ha="right")
    _save(writer, "f10_selective_coverage")
