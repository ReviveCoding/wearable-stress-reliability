import numpy as np

from healthsense.evaluation.metrics import episode_metrics, evaluate_predictions


def test_episode_metrics_resets_by_group():
    y = np.array([1, 1, 0, 1, 1, 0])
    p = np.array([0.9, 0.9, 0.1, 0.9, 0.9, 0.1])
    result = episode_metrics(
        y, p, threshold=0.5, hop_seconds=5, group_ids=np.array(["A"] * 3 + ["B"] * 3)
    )
    assert result["n_true_episodes"] == 2
    assert result["stress_episode_recall"] == 1.0


def test_metrics_returns_frontier():
    y = np.array([0, 1, 2, 1, 0, 2])
    p = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.2, 0.7, 0.1],
            [0.8, 0.1, 0.1],
            [0.1, 0.1, 0.8],
        ]
    )
    metrics, frontier = evaluate_predictions(
        y, p, hop_seconds=5, thresholds=[0.3, 0.5, 0.7], target_false_alert_minutes_per_hour=10
    )
    assert "macro_f1" in metrics
    assert len(frontier) == 3
