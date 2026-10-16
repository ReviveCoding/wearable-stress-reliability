import numpy as np

from healthsense.datasets.synthetic import make_synthetic_fixture
from healthsense.models import (
    MajorityNoAlert,
    MaskAwareTemporalEncoder,
    QualityAwareGBDT,
    RCPPFormer,
)
from healthsense.preprocessing.transforms import SequenceStandardizer


def _batch():
    batch = make_synthetic_fixture(n_subjects=6, windows_per_subject=12, sequence_length=8)
    scaler = SequenceStandardizer().fit(batch.sequence)
    batch.sequence = scaler.transform(batch.sequence)
    return batch


def test_c0_c1_probability_contract():
    batch = _batch()
    for model in [MajorityNoAlert(), QualityAwareGBDT(max_iter=10)]:
        model.fit(batch)
        pred = model.predict(batch)
        pred.validate(batch.n_samples, batch.n_classes)


def test_c2_c3_shared_prediction_contract():
    batch = _batch()
    c2 = MaskAwareTemporalEncoder(epochs=1, batch_size=16, hidden_dim=12).fit(batch)
    c3 = RCPPFormer(epochs=1, batch_size=16, hidden_dim=12).fit(batch)
    for model in [c2, c3]:
        pred = model.predict(batch)
        pred.validate(batch.n_samples, batch.n_classes)


def test_anchor_bank_excludes_self_reference_without_removing_other_anchors():
    import numpy as np

    from healthsense.anchors.bank import AnchorBank

    bank = AnchorBank(
        anchors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        anchor_labels=np.asarray([0, 1], dtype=np.int64),
        anchor_quality=np.asarray([1.0, 1.0], dtype=np.float32),
        selection_method="unit_test",
        source_indices=np.asarray([5, 9], dtype=np.int64),
    )
    expected, support = bank.expected(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        exclude_source_indices=np.asarray([5, 9], dtype=np.int64),
    )
    np.testing.assert_allclose(expected[0], [0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(expected[1], [1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(support, [1.0, 1.0], atol=1e-6)


def test_explicit_cuda_request_fails_closed_when_cuda_is_unavailable():
    import pytest
    import torch

    if torch.cuda.is_available():
        pytest.skip("CPU-only fail-closed behavior is not applicable on a CUDA runtime")
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        MaskAwareTemporalEncoder(device="cuda")._device()


def test_c3_cached_transforms_match_end_to_end_wrappers() -> None:
    from healthsense.config import load_config
    from healthsense.evaluation.calibration import TemperatureScaler
    from healthsense.pipeline import _c4_bundle, _c4_from_raw, _c5_bundle, _c5_from_raw

    batch = make_synthetic_fixture(n_subjects=6, windows_per_subject=12, sequence_length=8)
    scaler = SequenceStandardizer().fit(batch.sequence)
    batch.sequence = scaler.transform(batch.sequence)
    model = RCPPFormer(epochs=1, batch_size=16, hidden_dim=12).fit(batch)
    raw = model.predict(batch)
    temperature = TemperatureScaler().fit(raw.probabilities, batch.labels)
    config = load_config()

    np.testing.assert_allclose(
        _c4_from_raw(raw, batch, config).probabilities,
        _c4_bundle(model, batch, config).probabilities,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        _c5_from_raw(raw, temperature, batch).probabilities,
        _c5_bundle(model, temperature, batch).probabilities,
        rtol=1e-6,
        atol=1e-6,
    )
