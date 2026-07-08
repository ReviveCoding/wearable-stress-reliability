from __future__ import annotations

from importlib import resources
from pathlib import Path
import tomllib

import numpy as np
import pytest
import yaml

import healthsense
from healthsense.config import load_config, validate_config
from healthsense.datasets.synthetic import make_synthetic_fixture
from healthsense.preprocessing.split import ParticipantSplit


def test_checkout_import_precedes_any_ambient_editable_install() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    expected = (repo_root / "src" / "healthsense" / "__init__.py").resolve()
    assert Path(healthsense.__file__).resolve() == expected


def test_package_version_matches_pyproject() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    with (repo_root / "pyproject.toml").open("rb") as handle:
        metadata = tomllib.load(handle)
    assert healthsense.__version__ == metadata["project"]["version"]


def test_default_configuration_is_packaged_and_matches_repo_template() -> None:
    config = load_config()
    assert config["_config_path"] == "package://healthsense.resources/smoke.yaml"
    assert config["evaluation"]["threshold_grid"]
    repo_root = Path(__file__).resolve().parents[1]
    assert not (repo_root / "configs").exists()
    packaged = yaml.safe_load(
        resources.files("healthsense.resources").joinpath("smoke.yaml").read_text(encoding="utf-8")
    )
    assert packaged["models"]["device"] == "cpu"
    assert packaged["evaluation"]["release_gates"]["max_inference_p95_ms"] > 0


def test_invalid_configuration_fails_closed() -> None:
    config = load_config()
    config["evaluation"]["threshold_grid"] = [0.8, 0.2]
    with pytest.raises(ValueError, match="sorted ascending"):
        validate_config(config)


def test_canonical_contract_rejects_metadata_label_divergence() -> None:
    batch = make_synthetic_fixture(n_subjects=6, windows_per_subject=12, sequence_length=8)
    batch.metadata.loc[0, "state_label"] = int((batch.labels[0] + 1) % batch.n_classes)
    with pytest.raises(ValueError, match="state_label"):
        batch.validate()


def test_split_fails_when_a_role_lacks_a_class() -> None:
    batch = make_synthetic_fixture(n_subjects=6, windows_per_subject=12, sequence_length=8)
    subject_ids = sorted(batch.metadata["subject_id"].astype(str).unique().tolist())
    final_subject = subject_ids[0]
    final_idx = np.flatnonzero(batch.metadata["subject_id"].astype(str).to_numpy() == final_subject)
    batch.labels[final_idx] = 0
    batch.metadata.loc[final_idx, "state_label"] = 0
    batch.validate()
    remaining = [value for value in subject_ids if value != final_subject]
    split = ParticipantSplit(
        train_idx=np.flatnonzero(
            batch.metadata["subject_id"].astype(str).isin(remaining[:3]).to_numpy()
        ),
        calibration_idx=np.flatnonzero(
            batch.metadata["subject_id"].astype(str).to_numpy() == remaining[3]
        ),
        dev_idx=np.flatnonzero(batch.metadata["subject_id"].astype(str).to_numpy() == remaining[4]),
        final_idx=final_idx,
        manifest={},
    )
    with pytest.raises(ValueError, match="missing required classes"):
        split.validate(batch)


def test_invalid_channel_layout_fails_closed() -> None:
    config = load_config()
    config["data"]["n_channels"] = 7
    config["data"]["n_modalities"] = 4
    with pytest.raises(ValueError, match="divisible"):
        validate_config(config)


def test_nonfinite_configuration_fails_closed() -> None:
    config = load_config()
    config["models"]["learning_rate"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_config(config)


def test_missing_release_gate_fails_closed() -> None:
    config = load_config()
    del config["evaluation"]["release_gates"]["max_inference_p95_ms"]
    with pytest.raises(ValueError, match="missing required keys"):
        validate_config(config)


def test_runtime_overrides_are_validated_and_change_resolved_config_hash() -> None:
    from healthsense.pipeline import _runtime_config

    cpu = _runtime_config(None, output_root="outputs-a", run_id="run-a", device="cpu")
    auto = _runtime_config(None, output_root="outputs-a", run_id="run-a", device="auto")
    assert cpu["models"]["device"] == "cpu"
    assert auto["models"]["device"] == "auto"
    assert cpu["_config_hash"] != auto["_config_hash"]


def test_configuration_rejects_non_integral_counts_and_invalid_bootstrap() -> None:
    config = load_config()
    config["data"]["n_modalities"] = 3.5
    with pytest.raises(ValueError, match="positive integer"):
        validate_config(config)

    config = load_config()
    config["evaluation"]["bootstrap_resamples"] = 0
    with pytest.raises(ValueError, match="bootstrap_resamples"):
        validate_config(config)


def test_configuration_rejects_duplicate_thresholds_and_nan_policy() -> None:
    config = load_config()
    config["evaluation"]["threshold_grid"] = [0.2, 0.2, 0.5]
    with pytest.raises(ValueError, match="duplicate"):
        validate_config(config)

    config = load_config()
    config["policy"]["confidence_threshold"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_config(config)


def test_runtime_dependencies_include_tabulate_for_markdown_reports() -> None:
    import tomllib

    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    requirements = pyproject["project"]["dependencies"]
    assert any(requirement.lower().startswith("tabulate") for requirement in requirements)


def test_project_license_metadata_is_compatible_with_torch_constrained_setuptools() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    license_metadata = pyproject["project"]["license"]
    assert license_metadata == {"text": "MIT"}
    assert "license-files" not in pyproject["project"]
