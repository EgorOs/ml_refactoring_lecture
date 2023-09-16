from pathlib import Path

import pytest

from src.constants import PROJECT_ROOT


@pytest.fixture(name='sample_dataset_path')
def _sample_dataset_path() -> Path:
    return PROJECT_ROOT / 'tests' / 'assets' / 'sample_dataset'
