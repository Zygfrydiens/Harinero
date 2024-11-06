import pytest
import os
import json
from pathlib import Path


@pytest.fixture
def test_root_dir() -> Path:
    """Returns the root directory for tests"""
    return Path(__file__).parent


@pytest.fixture
def test_config(test_root_dir):
    """
    Loads test configuration from json and ensures paths are absolute
    """
    config_path = Path(__file__).parent.parent / 'configs' / 'test_config.json'
    with open(config_path) as f:
        config = json.load(f)

    config['paths']['database'] = str(test_root_dir / 'data' / 'test.db')
    config['paths']['media'] = str(test_root_dir / 'data' / 'media')

    return config
