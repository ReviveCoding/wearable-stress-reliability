from pathlib import Path

import numpy as np

from healthsense.datasets.synthetic import make_synthetic_fixture
from healthsense.evaluation.metrics import choose_operating_threshold, episode_metrics
from healthsense.models import (
    MaskAwareTemporalEncoder,
    RCPPFormer,
    TemporalAsymmetricBoundaryCorrection,
)
from healthsense.models.base import PredictionBundle
from healthsense.models.serialization import load_candidate_model, save_candidate_model
from healthsense.preprocessing.transforms import SequenceStandardizer


def test_episode_metrics_sorts_within_trace():
    # Input arrives time-shuffled. Time ordering, not input row ordering, defines the episode.
    y = np.array([1, 0, 1, 0])
    score = np.array([0.9, 0.1, 0.9, 0.1])
    result = episode_metrics(
        y,
        score,
        threshold=0.5,
        hop_seconds=5,
        group_ids=np.array(["A", "A", "A", "A"]),
        time_order=np.array([10.0, 5.0, 15.0, 20.0]),
    )
    # Chronological labels become [0, 1, 1, 0], one true episode.
    assert result["n_true_episodes"] == 1.0
    assert result["stress_episode_recall"] == 1.0


def test_tabc_cannot_bridge_trace_boundary():
    # The adjacent true predictions belong to different sessions, so no coverage inflation may occur.
    probs = np.array([[0.4, 0.6, 0.0], [0.55, 0.45, 0.0], [0.4, 0.6, 0.0]], dtype=np.float32)
    bundle = PredictionBundle(
        candidate_id="C3",
        probabilities=probs,
        predicted_labels=probs.argmax(axis=1),
        support_score=np.ones(3),
    )
    corrected = TemporalAsymmetricBoundaryCorrection(min_support=0.35).correct(
        bundle,
        np.ones(3),
        elevated_class=1,
        group_ids=np.array(["A", "B", "A"]),
        time_order=np.array([0.0, 0.0, 5.0]),
    )
    assert corrected.probabilities[1, 1] < 0.5


def test_dev_selection_does_not_require_final_frontier():
    frontier = __import__("pandas").DataFrame(
        {
            "threshold": [0.2, 0.5, 0.8],
            "stress_episode_recall": [1.0, 0.8, 0.5],
            "false_alert_minutes_per_nonstress_hour": [12.0, 5.0, 1.0],
        }
    )
    selected = choose_operating_threshold(frontier, target_minutes_per_hour=8.0)
    assert selected["threshold"] == 0.5
    assert selected["selection_status"] == "meets_dev_burden"


def test_neural_and_rcpp_roundtrip_preserve_prediction_contract(tmp_path: Path):
    batch = make_synthetic_fixture(n_subjects=6, windows_per_subject=12, sequence_length=8)
    scaler = SequenceStandardizer().fit(batch.sequence)
    batch.sequence = scaler.transform(batch.sequence)
    for model in [
        MaskAwareTemporalEncoder(epochs=1, batch_size=16, hidden_dim=12),
        RCPPFormer(epochs=1, batch_size=16, hidden_dim=12),
    ]:
        model.fit(batch)
        before = model.predict(batch).probabilities
        root = tmp_path / model.candidate_id
        save_candidate_model(model, root)

        # Exact probability round-trip is a same-runtime contract. The default loader is
        # intentionally portable and uses CPU; PyTorch does not guarantee bitwise-identical
        # floating-point output across CPU and CUDA. Use the fitted runtime for this equality
        # assertion, then separately verify that portable CPU loading preserves the prediction
        # bundle contract.
        if isinstance(model, MaskAwareTemporalEncoder):
            fitted_device = model.runtime_device_ or "cpu"
        else:
            assert model.encoder_ is not None
            fitted_device = model.encoder_.runtime_device_ or "cpu"

        after_same_runtime = (
            load_candidate_model(root, device=fitted_device).predict(batch).probabilities
        )
        np.testing.assert_allclose(before, after_same_runtime, rtol=1e-5, atol=1e-6)

        portable = load_candidate_model(root, device="cpu").predict(batch)
        portable.validate(batch.n_samples, batch.n_classes)
        assert np.isfinite(portable.probabilities).all()
        np.testing.assert_allclose(
            portable.probabilities.sum(axis=1),
            np.ones(batch.n_samples, dtype=np.float32),
            rtol=1e-5,
            atol=1e-6,
        )


def test_external_validation_small_fixture(tmp_path: Path):
    from healthsense.pipeline import run_external_validation

    def write_batch(root: Path, batch):
        np.savez_compressed(
            root / "batch.npz",
            sequence=batch.sequence,
            labels=batch.labels,
            modality_mask=batch.modality_mask,
            quality=batch.quality,
            timing_offset_ms=batch.timing_offset_ms,
            motion_artifact_score=batch.motion_artifact_score,
            channel_modalities=batch.channel_modalities,
        )
        batch.metadata.to_csv(root / "metadata.csv", index=False)

    source = make_synthetic_fixture(n_subjects=9, windows_per_subject=12, sequence_length=8)
    external = make_synthetic_fixture(
        seed=29, n_subjects=6, windows_per_subject=12, sequence_length=8
    )
    source_root = tmp_path / "source"
    external_root = tmp_path / "external"
    source_root.mkdir()
    external_root.mkdir()
    write_batch(source_root, source)
    write_batch(external_root, external)
    result = run_external_validation(
        source_root / "batch.npz",
        source_root / "metadata.csv",
        external_root / "batch.npz",
        external_root / "metadata.csv",
        output_root=tmp_path / "outputs",
        run_id="external-fixture",
    )
    assert (result / "external_validation_manifest.json").exists()
    assert (result / "candidate_comparison.csv").exists()


def test_immutable_split_manifest_roundtrip():
    from healthsense.preprocessing.split import create_participant_split, load_participant_split

    batch = make_synthetic_fixture(n_subjects=9, windows_per_subject=12, sequence_length=8)
    original = create_participant_split(
        batch, seed=17, final_holdout_subjects=2, calibration_subjects=1
    )
    restored = load_participant_split(batch, original.manifest)
    np.testing.assert_array_equal(original.final_idx, restored.final_idx)
    assert original.manifest["split_hash"] == restored.manifest["split_hash"]


def test_wesad_label_purity_exposes_transition_ambiguity():
    from healthsense.datasets.wesad import _window_label_purity

    label, purity = _window_label_purity(np.array([1] * 7 + [2] * 3 + [0] * 2))
    assert label == 1
    assert purity == 0.7


def test_cross_subject_manifest_roundtrip_preserves_fold_identity():
    from healthsense.preprocessing.split import (
        create_cross_subject_manifests,
        load_participant_split,
    )

    batch = make_synthetic_fixture(n_subjects=7, windows_per_subject=10, sequence_length=8)
    manifests = create_cross_subject_manifests(batch, seed=31, n_folds=3)
    assert len(manifests) == 3
    assert len({manifest["subjects"]["final"][0] for manifest in manifests}) == 3
    for manifest in manifests:
        restored = load_participant_split(batch, manifest)
        assert restored.manifest["fold_id"] == manifest["fold_id"]
        assert restored.manifest["split_hash"] == manifest["split_hash"]


def test_cli_exposes_cross_subject_confirmation_command():
    from healthsense.cli import build_parser

    args = build_parser().parse_args(
        [
            "cross-validate-canonical",
            "--npz",
            "data.npz",
            "--metadata",
            "metadata.csv",
            "--folds",
            "3",
        ]
    )
    assert args.command == "cross-validate-canonical"
    assert args.folds == 3


def test_external_binary_view_blocks_unknown_labels():
    from healthsense.pipeline import _binary_stress_view

    batch = make_synthetic_fixture(n_subjects=6, windows_per_subject=8, sequence_length=8)
    batch.labels[0] = -1
    try:
        _binary_stress_view(batch)
    except ValueError as exc:
        assert "reviewed non-negative labels" in str(exc)
    else:
        raise AssertionError("unknown labels must not be silently mapped to non-stress")


def test_synthetic_fixture_uses_subject_varying_episode_layouts():
    batch = make_synthetic_fixture(seed=67, n_subjects=6, windows_per_subject=12, sequence_length=8)
    layouts = []
    for _, frame in batch.metadata.groupby("subject_id", sort=True):
        layouts.append(tuple(frame.sort_values("window_end_s")["state_label"].tolist()))
    assert len(set(layouts)) > 1
    for layout in layouts:
        assert {0, 1, 2}.issubset(set(layout))


def test_external_transfer_rejects_feature_schema_mismatch():
    from healthsense.pipeline import _assert_external_compatible

    source = make_synthetic_fixture(n_subjects=6, windows_per_subject=12, sequence_length=8)
    external = make_synthetic_fixture(
        seed=51, n_subjects=6, windows_per_subject=12, sequence_length=8
    )
    external.metadata["feature_schema_id"] = "different_schema_v1"
    with __import__("pytest").raises(ValueError, match="feature schema mismatch"):
        _assert_external_compatible(source, external)
