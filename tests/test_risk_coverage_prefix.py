import numpy as np
import pytest

from healthsense.evaluation.metrics import risk_coverage_auc


@pytest.mark.parametrize("count", [0, 1, 2, 31, 257])
@pytest.mark.parametrize("tied", [False, True])
def test_risk_coverage_preserves_every_prefix(count, tied):
    rng = np.random.default_rng(51)
    probabilities = rng.dirichlet([1.0, 2.0, 1.0], size=count)
    if tied:
        probabilities[:] = [0.2, 0.6, 0.2]
    labels = rng.integers(0, 3, count)
    order = np.argsort(-probabilities.max(axis=1), kind="stable")
    correct = (probabilities.argmax(axis=1) == labels).astype(float)
    prefix_risks = [1.0 - correct[order[:size]].mean() for size in range(1, count + 1)]
    expected = float(np.trapezoid(prefix_risks, np.arange(1, count + 1) / count))
    assert risk_coverage_auc(labels, probabilities) == expected
