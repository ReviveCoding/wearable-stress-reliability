from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from healthsense.anchors.bank import AnchorBank, build_anchor_bank
from healthsense.config import load_config, with_overrides
from healthsense.datasets.contracts import CanonicalBatch
from healthsense.datasets.loaders import load_canonical
from healthsense.datasets.synthetic import make_synthetic_fixture
from healthsense.evaluation.calibration import TemperatureScaler
from healthsense.evaluation.metrics import (
    choose_operating_threshold,
    evaluate_predictions,
    paired_subject_bootstrap,
    selective_metrics,
    slice_metrics,
)
from healthsense.evaluation.robustness import run_robustness_suite
from healthsense.models import (
    MajorityNoAlert,
    MaskAwareTemporalEncoder,
    QualityAwareGBDT,
    RCPPFormer,
    TemporalAsymmetricBoundaryCorrection,
)
from healthsense.models.base import PredictionBundle
from healthsense.models.serialization import save_candidate_model
from healthsense.policy.decision import apply_selective_policy, release_decision
from healthsense.preprocessing.split import create_participant_split, load_participant_split
from healthsense.preprocessing.transforms import SequenceStandardizer
from healthsense.quality.gates import quality_dict
from healthsense.reporting.artifacts import ArtifactWriter
from healthsense.reporting.plots import generate_figures
from healthsense.utils.io import hardware_snapshot, stable_json_hash, tree_fingerprint, utc_now
from healthsense.utils.seed import set_global_seed


def _runtime_config(
    config_path: str | None,
    *,
    output_root: str | None = None,
    run_id: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Load and validate the fully resolved runtime configuration once.

    CLI/runtime overrides are included in the recorded config hash. This prevents a manifest
    from reporting a default CPU config when a caller actually requested CUDA, and avoids
    mutating a config after its reproducibility hash has been calculated.
    """
    config = load_config(config_path)
    overrides: dict[str, Any] = {}
    if output_root is not None:
        overrides["run.output_root"] = str(output_root)
    if run_id is not None:
        overrides["run.run_id"] = str(run_id)
    if device is not None:
        overrides["models.device"] = str(device).lower()
    return with_overrides(config, **overrides) if overrides else config


def _group_time(batch: CanonicalBatch) -> tuple[np.ndarray, np.ndarray]:
    group_ids = (
        batch.metadata["subject_id"].astype(str) + "::" + batch.metadata["session_id"].astype(str)
    ).to_numpy()
    time_order = batch.metadata["window_end_s"].to_numpy(dtype=float)
    return group_ids, time_order


def _eval_kwargs(
    config: dict[str, Any], batch: CanonicalBatch, *, operating_threshold: float | None = None
) -> dict[str, Any]:
    groups, times = _group_time(batch)
    return {
        "hop_seconds": config["data"]["hop_seconds"],
        "thresholds": config["evaluation"]["threshold_grid"],
        "target_false_alert_minutes_per_hour": config["evaluation"][
            "target_false_alert_minutes_per_hour"
        ],
        "elevated_class": config["policy"]["elevated_class"],
        "ece_bins": config["evaluation"]["ece_bins"],
        "group_ids": groups,
        "time_order": times,
        "operating_threshold": operating_threshold,
    }


def _apply_standardizer(batch: CanonicalBatch, scaler: SequenceStandardizer) -> CanonicalBatch:
    out = batch.subset(np.arange(batch.n_samples))
    out.sequence = scaler.transform(out.sequence)
    out.validate()
    return out


def _resolve_inference_device(requested: str) -> str:
    normalized = str(requested).lower()
    if normalized == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:  # pragma: no cover - torch is a runtime dependency
            return "cpu"
    return normalized


def _synchronize_if_cuda(device: str) -> None:
    if device != "cuda":
        return
    import torch

    torch.cuda.synchronize()


def _measure_predict(
    predict: Callable[[CanonicalBatch], PredictionBundle],
    batch: CanonicalBatch,
    *,
    device: str,
    repeats: int = 8,
) -> tuple[PredictionBundle, dict[str, float | int | str]]:
    """Measure end-to-end inference with CUDA synchronization when applicable.

    GPU kernels are asynchronous, so a timer without synchronize() materially under-reports
    latency. The metric is device-neutral and stores the actual device alongside p50/p95/p99.
    """
    resolved = _resolve_inference_device(device)
    _synchronize_if_cuda(resolved)
    bundle = predict(batch)  # warm-up before timing
    _synchronize_if_cuda(resolved)
    times: list[float] = []
    gpu_peak_mb = float("nan")
    if resolved == "cuda":
        import torch

        torch.cuda.reset_peak_memory_stats()
    for _ in range(repeats):
        _synchronize_if_cuda(resolved)
        start = time.perf_counter()
        bundle = predict(batch)
        _synchronize_if_cuda(resolved)
        times.append((time.perf_counter() - start) * 1000.0)
    if resolved == "cuda":
        import torch

        gpu_peak_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    return bundle, {
        "inference_p50_ms": float(np.quantile(times, 0.5)),
        "inference_p95_ms": float(np.quantile(times, 0.95)),
        "inference_p99_ms": float(np.quantile(times, 0.99)),
        "inference_device": resolved,
        "gpu_peak_memory_mb": gpu_peak_mb,
        "n_repeats": repeats,
    }


def _calibration_table(
    y_true: np.ndarray, probs: np.ndarray, candidate_id: str, bins: int = 10
) -> pd.DataFrame:
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(int)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf <= hi if hi == 1 else conf < hi)
        if mask.any():
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "bin_lower": lo,
                    "bin_upper": hi,
                    "count": int(mask.sum()),
                    "mean_confidence": float(conf[mask].mean()),
                    "empirical_accuracy": float(correct[mask].mean()),
                }
            )
    return pd.DataFrame(rows)


def _rcpp_ablation_features(
    rcpp: RCPPFormer,
    batch: CanonicalBatch,
    bundle: PredictionBundle,
    bank: AnchorBank,
    *,
    source_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build an anchor-ablation representation from one cached encoder output."""
    if bundle.embeddings is None:
        raise RuntimeError("missing encoder embedding")
    expected, anchor_support = bank.expected(
        bundle.embeddings, exclude_source_indices=source_indices
    )
    residual = bundle.embeddings - expected
    transition = (
        bundle.phase_probabilities
        if bundle.phase_probabilities is not None
        else np.zeros((batch.n_samples, 2), dtype=np.float32)
    )
    support = (
        bundle.support_score[:, None]
        if bundle.support_score is not None
        else np.ones((batch.n_samples, 1), dtype=np.float32)
    )
    quality_features = np.column_stack(
        [
            batch.quality.mean(axis=1),
            batch.modality_mask.mean(axis=1),
            batch.motion_artifact_score,
            rcpp._reliability(batch),
            support[:, 0],
            anchor_support,
        ]
    )
    return (
        np.concatenate(
            [bundle.embeddings, expected, residual, transition, quality_features], axis=1
        ).astype(np.float32),
        anchor_support.astype(np.float32),
    )


def _fit_rcpp_ablation(
    rcpp: RCPPFormer, train: CanonicalBatch, train_base: PredictionBundle, method: str
) -> tuple[AnchorBank, LogisticRegression]:
    """Fit one train-only anchor alternative once for both dev and final scoring."""
    if train_base.embeddings is None:
        raise RuntimeError("RCPP encoder did not produce train embeddings")
    bank = build_anchor_bank(
        train_base.embeddings,
        train.labels,
        rcpp._reliability(train),
        anchors_per_class=rcpp.anchors_per_class,
        method=method,
        seed=rcpp.seed,
    )
    x_train, _ = _rcpp_ablation_features(
        rcpp,
        train,
        train_base,
        bank,
        source_indices=np.arange(train.n_samples, dtype=np.int64),
    )
    classifier = LogisticRegression(
        C=0.6, max_iter=500, class_weight="balanced", random_state=rcpp.seed
    ).fit(x_train, train.labels)
    return bank, classifier


def _predict_rcpp_ablation(
    rcpp: RCPPFormer,
    target: CanonicalBatch,
    target_base: PredictionBundle,
    method: str,
    bank: AnchorBank,
    classifier: LogisticRegression,
) -> PredictionBundle:
    """Score a cached target encoder output with a previously fitted ablation alternative."""
    features, support = _rcpp_ablation_features(rcpp, target, target_base, bank)
    probs = classifier.predict_proba(features).astype(np.float32)
    bundle = PredictionBundle(
        candidate_id=f"C3_anchor_{method}",
        probabilities=probs,
        predicted_labels=probs.argmax(axis=1),
        embeddings=target_base.embeddings,
        support_score=support,
    )
    bundle.validate(target.n_samples, target.n_classes)
    return bundle


def _c5_from_raw(
    raw: PredictionBundle, temperature: TemperatureScaler, batch: CanonicalBatch
) -> PredictionBundle:
    """Apply the C5 calibration/routing transform to an already computed C3 prediction.

    Evaluation needs C3, C4, and C5 on the same batch. Reusing the immutable C3 output avoids
    repeated neural forward passes while preserving C4/C5's exact input contract. The public
    wrapper below remains intentionally end-to-end for latency and robustness measurement.
    """
    probs = temperature.transform(raw.probabilities)
    out = PredictionBundle(
        candidate_id="C5",
        probabilities=probs,
        predicted_labels=probs.argmax(axis=1),
        embeddings=raw.embeddings,
        phase_probabilities=raw.phase_probabilities,
        support_score=raw.support_score,
        rationale=[f"temperature={temperature.temperature_:.3f}" for _ in range(batch.n_samples)],
    )
    out.validate(batch.n_samples, batch.n_classes)
    return out


def _c5_bundle(
    c3: RCPPFormer, temperature: TemperatureScaler, batch: CanonicalBatch
) -> PredictionBundle:
    return _c5_from_raw(c3.predict(batch), temperature, batch)


def _c4_from_raw(
    raw: PredictionBundle, batch: CanonicalBatch, config: dict[str, Any]
) -> PredictionBundle:
    """Apply T-ABC to a cached C3 prediction without rerunning C3."""
    groups, times = _group_time(batch)
    return TemporalAsymmetricBoundaryCorrection().correct(
        raw,
        batch.quality.mean(axis=1),
        elevated_class=config["policy"]["elevated_class"],
        group_ids=groups,
        time_order=times,
    )


def _c4_bundle(c3: RCPPFormer, batch: CanonicalBatch, config: dict[str, Any]) -> PredictionBundle:
    return _c4_from_raw(c3.predict(batch), batch, config)


def _make_cards(
    writer: ArtifactWriter,
    *,
    run_id: str,
    dataset_label: str,
    data_quality: dict[str, Any],
    split: dict[str, Any],
    release: dict[str, Any],
) -> None:
    writer.text(
        "cards/data_card.md",
        f"# Data Card\n\nRun: `{run_id}`\n\nDataset run type: `{dataset_label}`. Raw WESAD and PhysioNet data are never redistributed by this repository.\n\n## Quality\n\n```json\n{data_quality}\n```\n\n## Split\n\n```json\n{split}\n```\n",
    )
    writer.text(
        "cards/model_card.md",
        "# Model Card\n\nCandidates C0-C5 are research evaluation candidates. C2 is a mask- and quality-gated temporal Transformer encoder. C3 is a PHT-derived Reference-Conditioned Physiological Phase residual model. C4 is optional ABC-inspired temporal boundary correction, and C5 applies calibration plus selective routing. None are clinical models.\n",
    )
    writer.text(
        "cards/evaluation_card.md",
        "# Evaluation Card\n\nPrimary metric: macro-F1. Primary decision metric: stress episode recall at a development-selected false-alert operating point. Thresholds are selected on a participant-disjoint development participant and frozen before final evaluation. Evaluation uses participant-disjoint split roles, calibration separation, quality/missingness stress tests, and artifact lineage.\n",
    )
    writer.text(
        "cards/release_card.md",
        f"# Release Card\n\n## Disposition\n\n`{release['disposition']}`\n\n## Rationale\n\n{release['reasons']}\n\nSynthetic smoke evidence is not public-data validation.\n",
    )
    writer.text(
        "reports/failure_taxonomy.md",
        "# Failure Taxonomy\n\n1. participant leakage\n2. development/final threshold leakage\n3. future-window leakage\n4. cross-session temporal correction\n5. modality dropout\n6. asynchronous modality lag\n7. motion artifact\n8. high-confidence unsupported decision\n9. episode under-coverage\n10. temporal overreach\n11. external-protocol shift\n12. exercise-as-stress confound\n",
    )


def _evaluate_candidates(
    *,
    config: dict[str, Any],
    dev: CanonicalBatch,
    final: CanonicalBatch,
    dev_bundles: dict[str, PredictionBundle],
    final_bundles: dict[str, PredictionBundle],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparison_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    frontier_rows: list[pd.DataFrame] = []
    calibration_rows: list[pd.DataFrame] = []
    slice_rows: list[pd.DataFrame] = []
    for cid in sorted(final_bundles):
        dev_metrics, dev_frontier = evaluate_predictions(
            dev.labels, dev_bundles[cid].probabilities, **_eval_kwargs(config, dev)
        )
        selection = choose_operating_threshold(
            dev_frontier, config["evaluation"]["target_false_alert_minutes_per_hour"]
        )
        threshold = float(selection["threshold"])
        final_metrics, final_frontier = evaluate_predictions(
            final.labels,
            final_bundles[cid].probabilities,
            **_eval_kwargs(config, final, operating_threshold=threshold),
        )
        comparison_rows.append({"candidate_id": cid, **final_metrics})
        selection_rows.append(
            {
                "candidate_id": cid,
                "selection_split": "participant_disjoint_dev",
                **selection,
                "dev_macro_f1": dev_metrics["macro_f1"],
                "dev_brier": dev_metrics["brier"],
            }
        )
        final_frontier = final_frontier.copy()
        final_frontier["candidate_id"] = cid
        final_frontier["frontier_split"] = "final_diagnostic_only_not_used_for_selection"
        frontier_rows.append(final_frontier)
        calibration_rows.append(
            _calibration_table(final.labels, final_bundles[cid].probabilities, cid)
        )
        sm = slice_metrics(
            final.metadata,
            final.labels,
            final_bundles[cid].probabilities,
            **_eval_kwargs(config, final, operating_threshold=threshold),
        )
        sm["candidate_id"] = cid
        slice_rows.append(sm)
    return (
        pd.DataFrame(comparison_rows).sort_values("candidate_id").reset_index(drop=True),
        pd.DataFrame(selection_rows).sort_values("candidate_id").reset_index(drop=True),
        pd.concat(frontier_rows, ignore_index=True),
        pd.concat(calibration_rows, ignore_index=True),
        pd.concat(slice_rows, ignore_index=True),
    )


def run_evaluation(
    batch: CanonicalBatch | None = None,
    config_path: str | None = None,
    output_root: str | None = None,
    run_id: str | None = None,
    dataset_label: str = "synthetic_fixture_only",
    split_manifest_path: str | Path | None = None,
    *,
    device: str | None = None,
    overwrite: bool = False,
) -> Path:
    config = _runtime_config(config_path, output_root=output_root, run_id=run_id, device=device)
    set_global_seed(int(config["run"]["seed"]), bool(config["run"].get("deterministic", True)))
    writer = ArtifactWriter.create(
        config["run"]["output_root"], config["run"]["run_id"], overwrite=overwrite
    )
    writer.json("config_resolved.json", config)

    if batch is None:
        batch = make_synthetic_fixture(
            seed=config["run"]["seed"],
            n_subjects=config["data"]["n_subjects"],
            windows_per_subject=config["data"]["windows_per_subject"],
            sequence_length=config["data"]["sequence_length"],
            n_modalities=config["data"]["n_modalities"],
            n_channels=config["data"]["n_channels"],
        )
        dataset_label = "synthetic_fixture_only"
    batch.validate()
    if np.any(batch.labels < 0):
        raise ValueError(
            "canonical evaluation requires non-negative state labels; PhysioNet tag maps must be verified first"
        )
    if int(config["policy"]["elevated_class"]) >= batch.n_classes:
        raise ValueError(
            "policy.elevated_class must be a valid class id for the evaluated canonical cohort"
        )
    observed_labels = np.unique(batch.labels)
    if not np.array_equal(observed_labels, np.arange(int(observed_labels.max()) + 1)):
        raise ValueError(
            "canonical supervised evaluation requires contiguous class ids starting at zero; "
            "review the source tag map and binary-label conversion before training"
        )
    if split_manifest_path is None:
        split = create_participant_split(
            batch,
            seed=config["run"]["seed"],
            final_holdout_subjects=config["data"]["final_holdout_subjects"],
            calibration_subjects=config["data"]["calibration_subjects"],
        )
    else:
        import json

        split = load_participant_split(
            batch, json.loads(Path(split_manifest_path).read_text(encoding="utf-8"))
        )
    data_quality = quality_dict(batch)
    writer.json(
        "data_fingerprint.json",
        {
            **batch.fingerprint_payload(),
            "fingerprint": stable_json_hash(batch.fingerprint_payload()),
        },
    )
    writer.json("split_manifest.json", split.manifest)
    writer.json("data_quality.json", data_quality)

    raw_train, raw_calib, raw_dev, raw_final = (
        batch.subset(split.train_idx),
        batch.subset(split.calibration_idx),
        batch.subset(split.dev_idx),
        batch.subset(split.final_idx),
    )
    scaler = SequenceStandardizer().fit(raw_train.sequence)
    train, calib, dev, final = (
        _apply_standardizer(x, scaler) for x in (raw_train, raw_calib, raw_dev, raw_final)
    )

    c0 = MajorityNoAlert().fit(train)
    c1 = QualityAwareGBDT(
        max_iter=config["models"]["c1_max_iter"], random_state=config["run"]["seed"]
    ).fit(train)
    c2 = MaskAwareTemporalEncoder(
        epochs=config["models"]["c2_epochs"],
        batch_size=config["models"]["batch_size"],
        learning_rate=config["models"]["learning_rate"],
        hidden_dim=config["models"]["hidden_dim"],
        dropout=config["models"]["dropout"],
        seed=config["run"]["seed"],
        device=config["models"].get("device", "cpu"),
    ).fit(train)
    c3 = RCPPFormer(
        epochs=config["models"]["c3_epochs"],
        batch_size=config["models"]["batch_size"],
        learning_rate=config["models"]["learning_rate"],
        hidden_dim=config["models"]["hidden_dim"],
        dropout=config["models"]["dropout"],
        seed=config["run"]["seed"],
        device=config["models"].get("device", "cpu"),
    ).fit(train)
    trained = {"C0": c0, "C1": c1, "C2": c2, "C3": c3}

    raw_c3_calib = c3.predict(calib)
    temperature = TemperatureScaler().fit(raw_c3_calib.probabilities, calib.labels)
    dev_bundles: dict[str, PredictionBundle] = {
        cid: model.predict(dev) for cid, model in trained.items()
    }
    final_bundles: dict[str, PredictionBundle] = {
        cid: model.predict(final) for cid, model in trained.items()
    }
    # C4 and C5 are deterministic transforms of C3 for ordinary evaluation. Reuse the C3
    # output instead of rerunning the temporal encoder; end-to-end wrappers remain in latency
    # and robustness paths, where the full path must be measured on each input.
    dev_bundles["C4"] = _c4_from_raw(dev_bundles["C3"], dev, config)
    final_bundles["C4"] = _c4_from_raw(final_bundles["C3"], final, config)
    dev_bundles["C5"] = _c5_from_raw(dev_bundles["C3"], temperature, dev)
    final_bundles["C5"] = _c5_from_raw(final_bundles["C3"], temperature, final)

    comparison, selections, frontier_df, calibration_df, slice_df = _evaluate_candidates(
        config=config,
        dev=dev,
        final=final,
        dev_bundles=dev_bundles,
        final_bundles=final_bundles,
    )
    selected_thresholds = dict(
        zip(selections["candidate_id"], selections["threshold"], strict=True)
    )

    operational_rows: list[dict[str, Any]] = []
    for cid, model in trained.items():
        _, op = _measure_predict(
            model.predict,
            final,
            device=(config["models"].get("device", "cpu") if cid in {"C2", "C3"} else "cpu"),
        )
        operational_rows.append(
            {
                "candidate_id": cid,
                **op,
                "training_seconds": getattr(model, "fit_seconds_", float("nan")),
                "model_family": type(model).__name__,
            }
        )
    _, c4_op = _measure_predict(
        lambda value: _c4_bundle(c3, value, config),
        final,
        device=config["models"].get("device", "cpu"),
    )
    operational_rows.append(
        {
            "candidate_id": "C4",
            **c4_op,
            "training_seconds": 0.0,
            "model_family": "TemporalAsymmetricBoundaryCorrection",
        }
    )
    _, c5_op = _measure_predict(
        lambda value: _c5_bundle(c3, temperature, value),
        final,
        device=config["models"].get("device", "cpu"),
    )
    operational_rows.append(
        {
            "candidate_id": "C5",
            **c5_op,
            "training_seconds": 0.0,
            "model_family": "TemperatureScaledSelectivePolicy",
        }
    )
    operational_df = pd.DataFrame(operational_rows)

    # Anchor ablation used to rebuild the same bank/classifier and rerun the encoder separately
    # for dev and final. Cache the train/dev/final encoder outputs and fit each anchor method once.
    # This preserves the train-only reference contract while reducing repeated neural inference.
    if c3.encoder_ is None:
        raise RuntimeError("RCPP encoder unavailable for anchor ablation")
    train_encoder_bundle = c3.encoder_.predict(train)
    dev_encoder_bundle = c3.encoder_.predict(dev)
    final_encoder_bundle = c3.encoder_.predict(final)
    ablation_rows = []
    for method in ["random", "nearest", "kmedoids", "reliability_weighted_medoids"]:
        # Each anchor alternative selects its own threshold on the same disjoint development
        # participant. Reusing C3's threshold would create an avoidable comparison bias.
        bank, classifier = _fit_rcpp_ablation(c3, train, train_encoder_bundle, method)
        dev_ablation = _predict_rcpp_ablation(c3, dev, dev_encoder_bundle, method, bank, classifier)
        final_ablation = _predict_rcpp_ablation(
            c3, final, final_encoder_bundle, method, bank, classifier
        )
        dev_metrics, dev_frontier = evaluate_predictions(
            dev.labels, dev_ablation.probabilities, **_eval_kwargs(config, dev)
        )
        selection = choose_operating_threshold(
            dev_frontier, config["evaluation"]["target_false_alert_minutes_per_hour"]
        )
        threshold = float(selection["threshold"])
        ab_metrics, _ = evaluate_predictions(
            final.labels,
            final_ablation.probabilities,
            **_eval_kwargs(config, final, operating_threshold=threshold),
        )
        ablation_rows.append(
            {
                "candidate_id": "C3",
                "anchor_method": method,
                "operating_threshold_source": "matching_anchor_method_dev",
                "operating_threshold": threshold,
                "selection_status": selection["selection_status"],
                "dev_macro_f1": dev_metrics["macro_f1"],
                "dev_brier": dev_metrics["brier"],
                **ab_metrics,
            }
        )
    ablation_df = pd.DataFrame(ablation_rows)

    robust_rows = []
    predictors: dict[str, Callable[[CanonicalBatch], PredictionBundle]] = {
        "C2": c2.predict,
        "C3": c3.predict,
        "C5": lambda value: _c5_bundle(c3, temperature, value),
    }
    for cid, predict in predictors.items():
        robust = run_robustness_suite(
            predict,
            final,
            evaluation_kwargs=_eval_kwargs(
                config, final, operating_threshold=float(selected_thresholds[cid])
            ),
        )
        robust["candidate_id"] = cid
        robust_rows.append(robust)
    robustness_df = pd.concat(robust_rows, ignore_index=True)

    scenario_rows = []
    scenario_defs = [
        ("conservative", 0.70, 0.50),
        (
            "standard",
            config["policy"]["confidence_threshold"],
            config["policy"]["quality_threshold"],
        ),
        ("safety_recall", 0.40, 0.25),
        ("review_capacity", 0.65, 0.45),
    ]
    policy_standard: pd.DataFrame | None = None
    for name, confidence_threshold, quality_threshold in scenario_defs:
        policy = apply_selective_policy(
            final_bundles["C5"].probabilities,
            final.quality,
            confidence_threshold=float(confidence_threshold),
            quality_threshold=float(quality_threshold),
            elevated_class=config["policy"]["elevated_class"],
        )
        metrics = selective_metrics(
            final.labels,
            final_bundles["C5"].probabilities,
            policy["defer"].to_numpy(dtype=bool),
            elevated_class=config["policy"]["elevated_class"],
        )
        scenario_rows.append(
            {
                "scenario": name,
                "confidence_threshold": float(confidence_threshold),
                "quality_threshold": float(quality_threshold),
                "elevated_support_rate": float(
                    (policy["action"] == "ELEVATED_STATE_SUPPORTED").mean()
                ),
                **metrics,
            }
        )
        if name == "standard":
            policy_standard = policy
    if policy_standard is None:
        raise RuntimeError("standard policy not constructed")
    decision_scenarios = pd.DataFrame(scenario_rows)
    release_selective = decision_scenarios.loc[decision_scenarios["scenario"] == "standard"].copy()
    release_selective.insert(0, "candidate_id", "C5")

    subject_summary = (
        slice_df[slice_df["candidate_id"].isin(["C2", "C3"])]
        .pivot_table(
            index="slice_value", columns="candidate_id", values="operating_stress_episode_recall"
        )
        .reset_index()
    )
    bootstrap = paired_subject_bootstrap(
        subject_summary.rename(columns={"C2": "baseline", "C3": "challenger"}),
        "baseline",
        "challenger",
        resamples=config["evaluation"]["bootstrap_resamples"],
        seed=config["run"]["seed"],
    )
    writer.json("statistical_comparison.json", {"C3_vs_C2_operating_episode_recall": bootstrap})

    release = release_decision(
        comparison,
        robustness_df[robustness_df["candidate_id"] == "C5"],
        operational_df,
        gates=config["evaluation"]["release_gates"],
        predictive_champion="C3",
        production_champion="C5",
        statistical_comparison=bootstrap,
        slice_comparison=subject_summary,
        selective=release_selective,
    )

    prediction_rows = final.metadata.copy()
    for cid, bundle in final_bundles.items():
        prediction_rows[f"{cid}_predicted_state"] = bundle.predicted_labels
        prediction_rows[f"{cid}_confidence"] = bundle.probabilities.max(axis=1)
        prediction_rows[f"{cid}_stress_probability"] = bundle.probabilities[
            :, config["policy"]["elevated_class"]
        ]
        prediction_rows[f"{cid}_operating_threshold_from_dev"] = float(selected_thresholds[cid])
        if bundle.support_score is not None:
            prediction_rows[f"{cid}_support"] = bundle.support_score
    prediction_rows = pd.concat([prediction_rows, policy_standard.add_prefix("C5_policy_")], axis=1)
    failure = prediction_rows.loc[
        (prediction_rows["C5_predicted_state"] != prediction_rows["state_label"])
        & (prediction_rows["C5_confidence"] >= prediction_rows["C5_confidence"].quantile(0.75))
    ].copy()

    writer.json(
        "metrics.json",
        {
            "candidates": comparison.to_dict(orient="records"),
            "synthetic_fixture_only": dataset_label == "synthetic_fixture_only",
        },
    )
    writer.table("candidate_comparison.csv", comparison)
    writer.table("operating_point_selection.csv", selections)
    writer.table("slice_metrics.csv", slice_df)
    writer.table("episode_metrics.csv", frontier_df)
    writer.table("robustness_metrics.csv", robustness_df)
    writer.table("calibration_table.csv", calibration_df)
    writer.json(
        "operational_metrics.json", {"candidates": operational_df.to_dict(orient="records")}
    )
    writer.table("operational_metrics.csv", operational_df)
    writer.table("decision_scenarios.csv", decision_scenarios)
    writer.table("selective_metrics.csv", release_selective)
    writer.table("predictions.csv", prediction_rows)
    writer.table("failure_cases.csv", failure)
    writer.table("ablations/rcpp_anchor_ablation.csv", ablation_df)
    writer.json("release_decision.json", release)
    writer.json(
        "calibration.json",
        {
            "candidate": "C5",
            "temperature": temperature.temperature_,
            "fit_split": "participant_disjoint_calibration",
        },
    )
    for cid, model in trained.items():
        save_candidate_model(model, writer.path(f"models/{cid}"))
    writer.track_tree("models")

    claim_boundary = (
        "This run is a deterministic synthetic fixture smoke. It cannot support wearable benchmark, external-transfer, clinical, or production claims."
        if dataset_label == "synthetic_fixture_only"
        else "This is a public-data canonical evaluation. It is non-clinical and cannot support diagnostic, treatment, Apple-device, or production-ready claims."
    )
    run_manifest = {
        "run_id": config["run"]["run_id"],
        "created_at_utc": utc_now(),
        "dataset": dataset_label,
        "dataset_fingerprint": stable_json_hash(batch.fingerprint_payload()),
        "split_manifest_hash": split.manifest["split_hash"],
        "config_hash": config["_config_hash"],
        "seed": config["run"]["seed"],
        "code_fingerprint": tree_fingerprint(Path(__file__).resolve().parent),
        "hardware": hardware_snapshot(),
        "threshold_selection": "participant-disjoint dev cohort only; final frontier is diagnostic only",
        "claim_boundary": claim_boundary,
    }
    writer.json("run_manifest.json", run_manifest)
    _make_cards(
        writer,
        run_id=config["run"]["run_id"],
        dataset_label=dataset_label,
        data_quality=data_quality,
        split=split.manifest,
        release=release,
    )
    writer.text(
        "reports/benchmark_report.md",
        f"# Benchmark Report\n\nDataset run type: `{dataset_label}`. Final-set operating metrics use thresholds selected on a participant-disjoint development cohort; final threshold frontiers are diagnostic only.\n\n"
        + comparison.to_markdown(index=False)
        + "\n\n## Release decision\n\n```json\n"
        + str(release)
        + "\n```\n",
    )
    generate_figures(
        writer,
        comparison=comparison,
        selections=selections,
        robustness=robustness_df,
        frontier=frontier_df,
        slice_df=slice_df,
        metadata=final.metadata,
        operational=operational_df,
        decision_scenarios=decision_scenarios,
        dataset_label=dataset_label,
    )
    writer.manifest()
    return writer.root


def run_smoke(
    config_path: str | None = None,
    output_root: str | None = None,
    run_id: str | None = None,
    *,
    device: str | None = None,
    overwrite: bool = False,
) -> Path:
    return run_evaluation(
        None,
        config_path,
        output_root,
        run_id,
        dataset_label="synthetic_fixture_only",
        device=device,
        overwrite=overwrite,
    )


def run_canonical(
    npz_path: str | Path,
    metadata_path: str | Path,
    config_path: str | None = None,
    output_root: str | None = None,
    run_id: str | None = None,
    split_manifest_path: str | Path | None = None,
    *,
    device: str | None = None,
    overwrite: bool = False,
) -> Path:
    batch = load_canonical(npz_path, metadata_path)
    dataset_values = sorted(batch.metadata["dataset_id"].astype(str).unique().tolist())
    return run_evaluation(
        batch,
        config_path,
        output_root,
        run_id,
        dataset_label="canonical:" + ",".join(dataset_values),
        split_manifest_path=split_manifest_path,
        device=device,
        overwrite=overwrite,
    )


def _binary_stress_view(batch: CanonicalBatch, *, force_nonstress: bool = False) -> CanonicalBatch:
    """Create an explicit binary research target without mutating source artifacts.

    WESAD amusement is mapped to non-stress only for the separate cross-protocol binary task.
    Exercise-confound inputs use `force_nonstress=True` so their outputs are measured as false
    stress support, never merged into the training target. Canonical batches with unknown labels
    are blocked: unknown PhysioNet windows must not silently become non-stress examples.
    """
    if np.any(np.asarray(batch.labels) < 0):
        raise ValueError(
            "external binary evaluation requires reviewed non-negative labels; prepare PhysioNet "
            "with a verified tag map before using stress/rest or exercise-confound evaluation"
        )
    out = batch.subset(np.arange(batch.n_samples))
    out.labels = (
        np.zeros(batch.n_samples, dtype=np.int64)
        if force_nonstress
        else (batch.labels == 1).astype(np.int64)
    )
    metadata = out.metadata.copy()
    metadata["state_label"] = out.labels
    transition = np.zeros(out.n_samples, dtype=np.int64)
    for _, group in metadata.groupby(["subject_id", "session_id"], sort=False):
        ordered = group.sort_values("window_end_s", kind="stable")
        labels = ordered["state_label"].to_numpy(dtype=int)
        positions = ordered.index.to_numpy(dtype=int)
        transition[positions[1:]] = (labels[1:] != labels[:-1]).astype(np.int64)
    metadata["transition_label"] = transition
    out.metadata = metadata
    out.validate()
    return out


def _feature_schema_id(batch: CanonicalBatch) -> str:
    if "feature_schema_id" not in batch.metadata.columns:
        raise ValueError(
            "external transfer requires feature_schema_id metadata; regenerate canonical artifacts "
            "with a schema-aware adapter"
        )
    values = sorted(batch.metadata["feature_schema_id"].dropna().astype(str).unique().tolist())
    if len(values) != 1 or not values[0] or values[0] == "nan":
        raise ValueError(
            "external transfer requires exactly one non-empty feature_schema_id per canonical artifact"
        )
    return values[0]


def _assert_external_compatible(source: CanonicalBatch, external: CanonicalBatch) -> None:
    if source.sequence.shape[1:] != external.sequence.shape[1:]:
        raise ValueError(
            f"external sequence shape mismatch: source {source.sequence.shape[1:]}, external {external.sequence.shape[1:]}"
        )
    if not np.array_equal(source.channel_modalities, external.channel_modalities):
        raise ValueError(
            "external channel_modalities mismatch; align sensor representation before transfer evaluation"
        )
    source_schema = _feature_schema_id(source)
    external_schema = _feature_schema_id(external)
    if source_schema != external_schema:
        raise ValueError(
            "external feature schema mismatch: "
            f"source={source_schema!r}, external={external_schema!r}; use a documented shared representation"
        )


def run_external_validation(
    source_npz: str | Path,
    source_metadata: str | Path,
    external_npz: str | Path,
    external_metadata: str | Path,
    *,
    external_mode: str = "stress_rest",
    config_path: str | None = None,
    output_root: str | None = None,
    run_id: str | None = None,
    device: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Run a locked cross-protocol confirmation or exercise-confound analysis.

    This path trains and chooses thresholds on source participants only. The external dataset is
    never used for training, calibration, threshold choice, candidate selection, or release
    advancement. It is confirmation evidence only.
    """
    if external_mode not in {"stress_rest", "exercise_confound"}:
        raise ValueError("external_mode must be stress_rest or exercise_confound")
    config = _runtime_config(config_path, output_root=output_root, run_id=run_id, device=device)
    set_global_seed(int(config["run"]["seed"]), bool(config["run"].get("deterministic", True)))
    writer = ArtifactWriter.create(
        config["run"]["output_root"], config["run"]["run_id"], overwrite=overwrite
    )
    writer.json("config_resolved.json", config)
    source_raw = load_canonical(source_npz, source_metadata)
    external_raw = load_canonical(external_npz, external_metadata)
    source = _binary_stress_view(source_raw)
    external = _binary_stress_view(
        external_raw, force_nonstress=external_mode == "exercise_confound"
    )
    if int(config["policy"]["elevated_class"]) >= source.n_classes:
        raise ValueError("policy.elevated_class must be valid for binary external evaluation")
    _assert_external_compatible(source, external)
    split = create_participant_split(
        source,
        seed=config["run"]["seed"],
        final_holdout_subjects=config["data"]["final_holdout_subjects"],
        calibration_subjects=config["data"]["calibration_subjects"],
    )
    raw_train, raw_calib, raw_dev = (
        source.subset(split.train_idx),
        source.subset(split.calibration_idx),
        source.subset(split.dev_idx),
    )
    scaler = SequenceStandardizer().fit(raw_train.sequence)
    train, calib, dev, final = (
        _apply_standardizer(value, scaler) for value in (raw_train, raw_calib, raw_dev, external)
    )
    c0 = MajorityNoAlert().fit(train)
    c1 = QualityAwareGBDT(
        max_iter=config["models"]["c1_max_iter"], random_state=config["run"]["seed"]
    ).fit(train)
    c2 = MaskAwareTemporalEncoder(
        epochs=config["models"]["c2_epochs"],
        batch_size=config["models"]["batch_size"],
        learning_rate=config["models"]["learning_rate"],
        hidden_dim=config["models"]["hidden_dim"],
        dropout=config["models"]["dropout"],
        seed=config["run"]["seed"],
        device=config["models"].get("device", "cpu"),
    ).fit(train)
    c3 = RCPPFormer(
        epochs=config["models"]["c3_epochs"],
        batch_size=config["models"]["batch_size"],
        learning_rate=config["models"]["learning_rate"],
        hidden_dim=config["models"]["hidden_dim"],
        dropout=config["models"]["dropout"],
        seed=config["run"]["seed"],
        device=config["models"].get("device", "cpu"),
    ).fit(train)
    trained = {"C0": c0, "C1": c1, "C2": c2, "C3": c3}
    temperature = TemperatureScaler().fit(c3.predict(calib).probabilities, calib.labels)
    dev_bundles = {cid: model.predict(dev) for cid, model in trained.items()}
    final_bundles = {cid: model.predict(final) for cid, model in trained.items()}
    dev_bundles["C4"] = _c4_from_raw(dev_bundles["C3"], dev, config)
    final_bundles["C4"] = _c4_from_raw(final_bundles["C3"], final, config)
    dev_bundles["C5"] = _c5_from_raw(dev_bundles["C3"], temperature, dev)
    final_bundles["C5"] = _c5_from_raw(final_bundles["C3"], temperature, final)
    comparison, selections, frontier, calibration, slices = _evaluate_candidates(
        config=config, dev=dev, final=final, dev_bundles=dev_bundles, final_bundles=final_bundles
    )
    prediction_rows = final.metadata.copy()
    for cid, bundle in final_bundles.items():
        prediction_rows[f"{cid}_stress_probability"] = bundle.probabilities[:, 1]
        prediction_rows[f"{cid}_predicted_state"] = bundle.predicted_labels
    mode_text = (
        "false-stress support under structured exercise"
        if external_mode == "exercise_confound"
        else "locked binary stress/rest cross-protocol confirmation"
    )
    writer.json(
        "external_validation_manifest.json",
        {
            "mode": external_mode,
            "purpose": mode_text,
            "source_fingerprint": stable_json_hash(source.fingerprint_payload()),
            "external_fingerprint": stable_json_hash(external.fingerprint_payload()),
            "source_split_hash": split.manifest["split_hash"],
            "selection_policy": "source dev cohort only",
            "external_role": "untouched confirmation; not a candidate-selection or release gate",
            "claim_boundary": "Non-clinical public-data confirmation only; not cross-device or production evidence.",
        },
    )
    writer.table("candidate_comparison.csv", comparison)
    writer.table("operating_point_selection.csv", selections)
    writer.table("episode_metrics.csv", frontier)
    writer.table("calibration_table.csv", calibration)
    writer.table("slice_metrics.csv", slices)
    writer.table("predictions.csv", prediction_rows)
    if external_mode == "exercise_confound":
        confound = pd.DataFrame(
            [
                {
                    "candidate_id": cid,
                    "exercise_false_stress_support_rate": float(
                        (bundle.predicted_labels == 1).mean()
                    ),
                    "mean_stress_probability": float(bundle.probabilities[:, 1].mean()),
                }
                for cid, bundle in final_bundles.items()
            ]
        )
        writer.table("exercise_confound_metrics.csv", confound)
    writer.text(
        "reports/external_validation_report.md",
        f"# External Validation Report\n\nMode: `{external_mode}`. Purpose: {mode_text}. Thresholds were selected on source development participants only. This external result cannot select a champion or advance a release.\n\n"
        + comparison.to_markdown(index=False)
        + "\n",
    )
    for cid, model in trained.items():
        save_candidate_model(model, writer.path(f"models/{cid}"))
    writer.track_tree("models")
    writer.manifest()
    return writer.root


def run_cross_subject_confirmation(
    npz_path: str | Path,
    metadata_path: str | Path,
    *,
    n_folds: int = 5,
    config_path: str | None = None,
    output_root: str | None = None,
    run_id: str | None = None,
    device: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Run deterministic nested participant-held-out confirmation folds.

    Each fold selects thresholds on a development participant and evaluates one untouched final
    participant. The aggregate report is evidence for stability, while each fold remains an
    independently reproducible artifact package.
    """
    import json

    from healthsense.preprocessing.split import create_cross_subject_manifests

    config = _runtime_config(config_path, output_root=output_root, run_id=run_id, device=device)
    batch = load_canonical(npz_path, metadata_path)
    manifests = create_cross_subject_manifests(batch, seed=config["run"]["seed"], n_folds=n_folds)
    writer = ArtifactWriter.create(
        config["run"]["output_root"], config["run"]["run_id"], overwrite=overwrite
    )
    writer.json("config_resolved.json", config)
    writer.json("cross_subject_plan.json", {"n_folds": n_folds, "folds": manifests})
    fold_rows = []
    for manifest in manifests:
        fold_id = int(manifest["fold_id"])
        manifest_path = writer.json(f"fold_manifests/fold_{fold_id:02d}.json", manifest)
        fold_root = run_evaluation(
            batch,
            config_path,
            str(writer.path("folds")),
            f"fold-{fold_id:02d}",
            dataset_label="canonical_cross_subject_confirmation",
            split_manifest_path=manifest_path,
            device=device,
            overwrite=False,
        )
        comparison = pd.read_csv(fold_root / "candidate_comparison.csv")
        comparison["fold_id"] = fold_id
        comparison["final_subject"] = manifest["subjects"]["final"][0]
        fold_rows.append(comparison)
    fold_comparison = pd.concat(fold_rows, ignore_index=True)
    summary = (
        fold_comparison.groupby("candidate_id", as_index=False)
        .agg(
            n_folds=("fold_id", "nunique"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            operating_episode_recall_mean=("operating_stress_episode_recall", "mean"),
            operating_episode_recall_std=("operating_stress_episode_recall", "std"),
            brier_mean=("brier", "mean"),
            adaptive_ece_mean=("adaptive_ece", "mean"),
        )
        .sort_values("candidate_id")
    )
    pivot = fold_comparison.pivot_table(
        index="fold_id", columns="candidate_id", values="operating_stress_episode_recall"
    ).reset_index()
    bootstrap = paired_subject_bootstrap(
        pivot.rename(columns={"C2": "baseline", "C3": "challenger"}),
        "baseline",
        "challenger",
        resamples=config["evaluation"]["bootstrap_resamples"],
        seed=config["run"]["seed"],
    )
    evidence_status = (
        "C3 positive subject-held-out episode-recall uplift supported"
        if bootstrap["n_subjects"] >= 5 and bootstrap["ci_low"] > 0
        else "No superiority claim supported; inspect fold artifacts and failure modes"
    )
    writer.table("fold_candidate_comparison.csv", fold_comparison)
    writer.table("cross_subject_summary.csv", summary)
    writer.json(
        "cross_subject_statistical_comparison.json",
        {"C3_vs_C2_operating_episode_recall": bootstrap, "evidence_status": evidence_status},
    )
    writer.text(
        "reports/cross_subject_confirmation_report.md",
        "# Cross-Subject Confirmation Report\n\n"
        f"Folds: `{n_folds}`. Each fold uses a unique untouched final participant and a separate calibration/dev participant.\n\n"
        + summary.to_markdown(index=False)
        + "\n\n## C3 vs C2 participant bootstrap\n\n```json\n"
        + json.dumps(bootstrap, indent=2)
        + "\n```\n\n"
        + evidence_status
        + "\n",
    )
    writer.track_tree("folds")
    writer.manifest()
    return writer.root
