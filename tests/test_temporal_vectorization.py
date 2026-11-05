import numpy as np
import pandas as pd
import pytest

from healthsense.models.tabc import TemporalAsymmetricBoundaryCorrection
from healthsense.temporal import ordered_trace_indices


def reference_correction(raw, support, quality, max_gap, minimum):
    result = raw.copy()
    for i in range(1, len(raw) - 1):
        if (
            not raw[i]
            and raw[i - 1]
            and raw[i + 1]
            and max_gap >= 1
            and support[i] >= minimum
            and quality[i] >= minimum
        ):
            result[i] = True
    for i in range(len(raw)):
        if (
            result[i]
            and not (i > 0 and result[i - 1])
            and not (i + 1 < len(raw) and result[i + 1])
            and (support[i] < minimum or quality[i] < minimum)
        ):
            result[i] = False
    return result


@pytest.mark.parametrize("n", [0, 1, 2, 3, 41, 257])
@pytest.mark.parametrize("gap", [0, 1, 3])
@pytest.mark.parametrize("dtype", [bool, int, float])
def test_temporal_correction_keeps_inflation_and_deflation(n, gap, dtype):
    rng = np.random.default_rng(51)
    raw = rng.integers(0, 2, n).astype(dtype)
    support, quality = rng.random((2, n))
    if n > 5:
        support[3] = np.nan
        quality[5] = np.nan
    expected = reference_correction(raw, support, quality, gap, 0.35)
    actual = TemporalAsymmetricBoundaryCorrection._correct_trace(raw, support, quality, gap, 0.35)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype


@pytest.mark.parametrize("count", [0, 1, 97])
def test_trace_groups_keep_first_seen_order_and_stable_time_ties(count):
    groups = np.resize(np.array(["b", "a", "nan", "b", "None"]), count)
    ordering = np.arange(count) % 3
    expected = []
    for key in pd.Series(groups).drop_duplicates():
        positions = np.flatnonzero(groups == key)
        expected.append(positions[np.argsort(ordering[positions], kind="stable")])
    actual = ordered_trace_indices(groups, ordering)
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(left, right)
