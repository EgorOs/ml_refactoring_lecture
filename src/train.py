import os

import lightning
from clearml import Task
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from src.callbacks.experiment_tracking import LogConfusionMatrix
from src.config import ExperimentConfig
from src.constants import PROJECT_ROOT
from src.datamodule import ClassificationDataModule
from src.lightning_module import ClassificationLightningModule


def train(cfg: ExperimentConfig):
    lightning.seed_everything(0)

    Task.force_requirements_env_freeze()  # or use task.set_packages() for more granular control.
    task = Task.init(
        project_name=cfg.project_name,
        task_name=cfg.experiment_name,
        # If `output_uri=True` uses default ClearML output URI,
        # can use string value to specify custom storage URI like S3.
        output_uri=True,
    )
    datamodule = ClassificationDataModule(cfg=cfg.data_config)

    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(save_top_k=3, monitor='valid_f1', mode='max', every_n_epochs=1),
    ]
    if cfg.track_in_clearml:
        callbacks += [LogConfusionMatrix(task, datamodule.idx_to_class)]
    model = ClassificationLightningModule(class_to_idx=datamodule.class_to_idx)
    trainer = Trainer(**dict(cfg.trainer_config), callbacks=callbacks, overfit_batches=60)
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)


if __name__ == '__main__':
    cfg_path = os.getenv('TRAIN_CFG_PATH', PROJECT_ROOT / 'configs' / 'train.yaml')
    train(cfg=ExperimentConfig.from_yaml(cfg_path))
