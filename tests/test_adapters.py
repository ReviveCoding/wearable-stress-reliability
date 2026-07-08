from pathlib import Path

import numpy as np
import pandas as pd

from healthsense.datasets.physionet import prepare_physionet
from healthsense.datasets.wesad import prepare_wesad


def _write_e4(path: Path, rate: int, data: np.ndarray) -> None:
    rows = [[1000.0], [rate]]
    rows.extend(data.tolist())
    pd.DataFrame(rows).to_csv(path, header=False, index=False)


def test_wesad_requires_explicit_pickle_acknowledgement(tmp_path: Path):
    with __import__("pytest").raises(ValueError, match="trust_pickle"):
        prepare_wesad(tmp_path, tmp_path / "out", trust_pickle=False)


def test_physionet_converter_uses_tag_map_for_labels(tmp_path: Path):
    subject = tmp_path / "STRESS" / "S01"
    subject.mkdir(parents=True)
    _write_e4(subject / "BVP.csv", 64, np.ones((64 * 70, 1)))
    _write_e4(subject / "EDA.csv", 4, np.ones((4 * 70, 1)))
    _write_e4(subject / "TEMP.csv", 4, np.ones((4 * 70, 1)))
    _write_e4(subject / "ACC.csv", 32, np.ones((32 * 70, 3)))
    pd.DataFrame([[1000.0], [1040.0], [1070.0]]).to_csv(
        subject / "tags.csv", header=False, index=False
    )
    tag_map = tmp_path / "tag_map.csv"
    pd.DataFrame(
        [
            {
                "session_family": "STRESS",
                "subject_id": "S01",
                "start_tag_index": 0,
                "end_tag_index": 2,
                "state_label": 1,
            }
        ]
    ).to_csv(tag_map, index=False)
    out = tmp_path / "prepared"
    manifest = prepare_physionet(
        tmp_path, out, tag_map_path=tag_map, window_seconds=30, hop_seconds=10, sequence_length=8
    )
    assert manifest["n_windows"] > 0
    assert manifest["labeled_windows"] > 0
    assert (out / "physionet_canonical_windows.npz").exists()


def test_wesad_converter_produces_labeled_canonical_windows(tmp_path: Path):
    import pickle

    rng = np.random.default_rng(7)
    subject = tmp_path / "S2"
    subject.mkdir(parents=True)
    duration_seconds = 30
    payload = {
        "signal": {
            "wrist": {
                "ACC": rng.normal(size=(32 * duration_seconds, 3)).astype(np.float32),
                "BVP": rng.normal(size=(64 * duration_seconds, 1)).astype(np.float32),
                "EDA": rng.normal(size=(4 * duration_seconds, 1)).astype(np.float32),
                "TEMP": rng.normal(size=(4 * duration_seconds, 1)).astype(np.float32),
            }
        },
        "label": np.repeat(np.array([1, 2, 3], dtype=np.int64), 700 * 10),
    }
    with (subject / "S2.pkl").open("wb") as handle:
        pickle.dump(payload, handle)
    out = tmp_path / "prepared"
    manifest = prepare_wesad(
        tmp_path,
        out,
        trust_pickle=True,
        sensor_config="wrist",
        window_seconds=10,
        hop_seconds=10,
        sequence_length=8,
        min_label_purity=1.0,
    )
    assert manifest["batch"]["n_samples"] == 3
    metadata = pd.read_csv(out / "wesad_window_metadata.csv")
    assert metadata["state_label"].tolist() == [0, 1, 2]


def test_physionet_unreviewed_windows_are_preserved_but_blocked_from_supervised_run(tmp_path: Path):
    from healthsense.datasets.loaders import load_canonical
    from healthsense.pipeline import run_canonical

    subject = tmp_path / "STRESS" / "S01"
    subject.mkdir(parents=True)
    _write_e4(subject / "BVP.csv", 64, np.ones((64 * 70, 1)))
    _write_e4(subject / "EDA.csv", 4, np.ones((4 * 70, 1)))
    _write_e4(subject / "TEMP.csv", 4, np.ones((4 * 70, 1)))
    _write_e4(subject / "ACC.csv", 32, np.ones((32 * 70, 3)))
    pd.DataFrame([[1000.0], [1040.0], [1070.0]]).to_csv(
        subject / "tags.csv", header=False, index=False
    )
    out = tmp_path / "prepared"
    manifest = prepare_physionet(
        tmp_path, out, tag_map_path=None, window_seconds=30, hop_seconds=10, sequence_length=8
    )
    assert manifest["unlabeled_windows"] > 0
    batch = load_canonical(
        out / "physionet_canonical_windows.npz", out / "physionet_window_metadata.csv"
    )
    assert batch.has_unknown_labels
    with __import__("pytest").raises(ValueError, match="non-negative state labels"):
        run_canonical(
            out / "physionet_canonical_windows.npz",
            out / "physionet_window_metadata.csv",
            output_root=tmp_path / "outputs",
            run_id="unreviewed",
        )


def test_wesad_shared_representation_is_explicit_and_chest_wrist_compatible(tmp_path: Path):
    import pickle

    rng = np.random.default_rng(9)
    subject = tmp_path / "S2"
    subject.mkdir(parents=True)
    duration_seconds = 30
    payload = {
        "signal": {
            "wrist": {
                "ACC": rng.normal(size=(32 * duration_seconds, 3)).astype(np.float32),
                "BVP": rng.normal(size=(64 * duration_seconds, 1)).astype(np.float32),
                "EDA": rng.normal(size=(4 * duration_seconds, 1)).astype(np.float32),
                "TEMP": rng.normal(size=(4 * duration_seconds, 1)).astype(np.float32),
            },
            "chest": {
                "ACC": rng.normal(size=(700 * duration_seconds, 3)).astype(np.float32),
                "ECG": rng.normal(size=(700 * duration_seconds, 1)).astype(np.float32),
                "EDA": rng.normal(size=(700 * duration_seconds, 1)).astype(np.float32),
                "EMG": rng.normal(size=(700 * duration_seconds, 1)).astype(np.float32),
                "Resp": rng.normal(size=(700 * duration_seconds, 1)).astype(np.float32),
                "Temp": rng.normal(size=(700 * duration_seconds, 1)).astype(np.float32),
            },
        },
        "label": np.repeat(np.array([1, 2, 3], dtype=np.int64), 700 * 10),
    }
    with (subject / "S2.pkl").open("wb") as handle:
        pickle.dump(payload, handle)
    shared_wrist = prepare_wesad(
        tmp_path,
        tmp_path / "wrist",
        trust_pickle=True,
        sensor_config="wrist",
        representation="shared_eda_temp_acc",
        window_seconds=10,
        hop_seconds=10,
        sequence_length=8,
        min_label_purity=1.0,
    )
    shared_chest = prepare_wesad(
        tmp_path,
        tmp_path / "chest",
        trust_pickle=True,
        sensor_config="chest",
        representation="shared_eda_temp_acc",
        window_seconds=10,
        hop_seconds=10,
        sequence_length=8,
        min_label_purity=1.0,
    )
    assert shared_wrist["feature_schema_id"] == "wesad_shared_eda_temp_acc_v1"
    assert shared_chest["feature_schema_id"] == "wesad_shared_eda_temp_acc_v1"
    assert shared_wrist["canonical_modalities"] == ["EDA", "TEMP", "ACC"]
    assert shared_chest["canonical_modalities"] == ["EDA", "TEMP", "ACC"]
