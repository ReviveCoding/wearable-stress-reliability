import numpy as np
import pytest

from healthsense.datasets.synthetic import make_synthetic_fixture


def test_synthetic_contract_valid():
    batch = make_synthetic_fixture(n_subjects=6, windows_per_subject=12, sequence_length=8)
    batch.validate()
    assert batch.n_samples == 72
    assert batch.as_feature_frame().shape[0] == 72


def test_contract_rejects_nonfinite():
    batch = make_synthetic_fixture(n_subjects=6, windows_per_subject=12, sequence_length=8)
    batch.sequence[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        batch.validate()


def test_prepared_unknown_labels_are_valid_but_n_classes_is_blocked():
    batch = make_synthetic_fixture(n_subjects=6, windows_per_subject=12, sequence_length=8)
    batch.labels[0] = -1
    batch.metadata.loc[0, "state_label"] = -1
    batch.validate()
    assert batch.has_unknown_labels
    with pytest.raises(ValueError, match="n_classes is undefined"):
        _ = batch.n_classes


def test_contract_rejects_nonfinite_quality_and_noninteger_channel_map():
    batch = make_synthetic_fixture(n_subjects=6, windows_per_subject=12, sequence_length=8)
    batch.quality[0, 0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        batch.validate()
    batch = make_synthetic_fixture(n_subjects=6, windows_per_subject=12, sequence_length=8)
    batch.channel_modalities = batch.channel_modalities.astype(np.float32)
    with pytest.raises(ValueError, match="integer dtype"):
        batch.validate()


def test_contract_rejects_nonbinary_mask_and_out_of_range_motion():
    batch = make_synthetic_fixture(n_subjects=6, windows_per_subject=12, sequence_length=8)
    batch.modality_mask[0, 0] = 0.5
    with pytest.raises(ValueError, match="binary"):
        batch.validate()
    batch = make_synthetic_fixture(n_subjects=6, windows_per_subject=12, sequence_length=8)
    batch.motion_artifact_score[0] = 1.5
    with pytest.raises(ValueError, match="motion_artifact_score"):
        batch.validate()
