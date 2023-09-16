import os

import lightning
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from src.callbacks.experiment_tracking import (
    ClearMLTracking,
    LogConfusionMatrix,
)
from src.config import ExperimentConfig
from src.constants import PROJECT_ROOT
from src.datamodule import ClassificationDataModule
from src.lightning_module import ClassificationLightningModule


def train(cfg: ExperimentConfig):
    lightning.seed_everything(0)
    datamodule = ClassificationDataModule(cfg=cfg.data_config)

    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(save_top_k=3, monitor='valid_f1', mode='max', every_n_epochs=1),
    ]
    if cfg.track_in_clearml:
        tracking_cb = ClearMLTracking(cfg, label_enumeration=datamodule.class_to_idx)
        callbacks += [tracking_cb, LogConfusionMatrix(tracking_cb, datamodule.idx_to_class)]
    model = ClassificationLightningModule(class_to_idx=datamodule.class_to_idx)

    trainer = Trainer(**dict(cfg.trainer_config), callbacks=callbacks, overfit_batches=60)
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)


if __name__ == '__main__':
    cfg_path = os.getenv('TRAIN_CFG_PATH', PROJECT_ROOT / 'configs' / 'train.yaml')
    train(cfg=ExperimentConfig.from_yaml(cfg_path))
