"""Configuration seed validation must not treat bool as an integer."""

import pytest

from healthsense.config import load_config, validate_config, with_overrides


def test_default_config_uses_valid_integer_seed():
    config = load_config()
    assert type(config["run"]["seed"]) is int
    validate_config(config)


def test_boolean_seed_is_rejected_before_hashing():
    config = load_config()
    with pytest.raises(ValueError, match="run.seed"):
        with_overrides(config, **{"run.seed": True})
