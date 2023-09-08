import os

import lightning
from lightning import Trainer

from src.config import ExperimentConfig
from src.constants import PROJECT_ROOT
from src.datamodule import ClassificationDataModule
from src.lightning_module import ClassificationLightningModule


def train(cfg: ExperimentConfig):
    lightning.seed_everything(0)
    datamodule = ClassificationDataModule(cfg=cfg.data_config)

    model = ClassificationLightningModule(class_to_idx=datamodule.class_to_idx)
    trainer = Trainer(**dict(cfg.trainer_config), callbacks=[], overfit_batches=60)
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)


if __name__ == '__main__':
    cfg_path = os.getenv('TRAIN_CFG_PATH', PROJECT_ROOT / 'configs' / 'train.yaml')
    train(cfg=ExperimentConfig.from_yaml(cfg_path))
