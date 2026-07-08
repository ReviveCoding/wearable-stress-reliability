import numpy as np

from healthsense.datasets.synthetic import make_synthetic_fixture
from healthsense.preprocessing.split import create_participant_split


def test_split_is_participant_disjoint_and_deterministic():
    batch = make_synthetic_fixture(n_subjects=9, windows_per_subject=12, sequence_length=8)
    a = create_participant_split(batch, seed=17, final_holdout_subjects=2, calibration_subjects=1)
    b = create_participant_split(batch, seed=17, final_holdout_subjects=2, calibration_subjects=1)
    assert np.array_equal(a.train_idx, b.train_idx)
    a.validate(batch)
    assert len(set(a.manifest["subjects"]["train"]) & set(a.manifest["subjects"]["final"])) == 0
