from pathlib import Path
from unittest.mock import Mock, patch

from src.config import ExperimentConfig
from src.constants import PROJECT_ROOT
from src.datamodule import ClearmlDataset
from src.train import train


@patch('src.datamodule.ClearmlDataset')
def test_training_pipeline(clearml_dataset: Mock, sample_dataset_path: Path):
    # Given
    local_dataset = Mock(ClearmlDataset)
    local_dataset.get_local_copy.return_value = sample_dataset_path
    clearml_dataset.get.return_value = local_dataset

    cfg_path = PROJECT_ROOT / 'configs' / 'train.yaml'
    train_cfg = ExperimentConfig.from_yaml(cfg_path)

    train_cfg.trainer_config.fast_dev_run = True
    train_cfg.data_config.batch_size = 2
    train_cfg.track_in_clearml = False

    # When
    train(cfg=train_cfg)
