import numpy as np

from healthsense.datasets.synthetic import make_synthetic_fixture
from healthsense.quality.gates import apply_modality_dropout, inject_timing_lag


def test_dropout_zeros_only_selected_modality():
    batch = make_synthetic_fixture(n_subjects=6, windows_per_subject=12, sequence_length=8)
    dropped = apply_modality_dropout(batch, 1)
    channels = dropped.channel_modalities == 1
    assert np.all(dropped.sequence[:, :, channels] == 0)
    assert np.all(dropped.modality_mask[:, 1] == 0)


def test_lag_preserves_causal_window_shape():
    batch = make_synthetic_fixture(n_subjects=6, windows_per_subject=12, sequence_length=8)
    lagged = inject_timing_lag(batch, 0, 5000)
    assert lagged.sequence.shape == batch.sequence.shape
    assert np.all(lagged.timing_offset_ms[:, 0] >= batch.timing_offset_ms[:, 0])
