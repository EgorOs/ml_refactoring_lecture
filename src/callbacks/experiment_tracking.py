import os
from typing import Dict, List, Optional

import torch
from clearml import OutputModel, Task
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from torch import Tensor

from src.config import ExperimentConfig
from src.logger import LOGGER


class ClearMLTracking(Callback):
    def __init__(
        self,
        cfg: ExperimentConfig,
        label_enumeration: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.label_enumeration = label_enumeration
        self.task: Optional[Task] = None
        self.output_model: Optional[OutputModel] = None

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._setup_task()

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        final_checkpoint = select_checkpoint_for_export(trainer)
        LOGGER.info('Uploading checkpoint "%s" to ClearML', final_checkpoint)
        self.output_model.update_weights(weights_filename=final_checkpoint, auto_delete_file=True)

    def _setup_task(self):
        Task.force_requirements_env_freeze()  # or use task.set_packages() for more granular control.
        self.task = Task.init(
            project_name=self.cfg.project_name,
            task_name=self.cfg.experiment_name,
            # If `output_uri=True` uses default ClearML output URI,
            # can use string value to specify custom storage URI like S3.
            output_uri=True,
        )
        self.task.connect_configuration(configuration=self.cfg.model_dump())
        self.output_model = OutputModel(task=self.task, label_enumeration=self.label_enumeration)


class LogConfusionMatrix(Callback):
    def __init__(self, tracking_cb: ClearMLTracking, idx_to_class: Dict[int, str]):
        super().__init__()
        self.tracking_cb = tracking_cb
        self.idx_to_class = idx_to_class
        self.predicts: List[Tensor] = []
        self.targets: List[Tensor] = []

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor,
        batch: List[Tensor],
        batch_idx: int,
    ) -> None:
        self.predicts.append(outputs)
        self.targets.append(batch[1])

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        targets = torch.cat(self.targets, dim=0).detach().cpu().numpy()
        predicts = torch.cat(self.predicts, dim=0).detach().cpu().numpy()
        cf_matrix = confusion_matrix(targets, predicts)
        cf_labels = [self.idx_to_class[idx] for idx, _ in enumerate(cf_matrix)]
        self.tracking_cb.task.logger.report_confusion_matrix(
            'Confusion matrix',
            'ignored',
            iteration=trainer.global_step,
            matrix=cf_matrix,
            xaxis=None,
            yaxis=None,
            xlabels=cf_labels,
            ylabels=cf_labels,
        )


def select_checkpoint_for_export(trainer: Trainer) -> str:
    checkpoint_cb: Optional[ModelCheckpoint] = trainer.checkpoint_callback
    if checkpoint_cb is not None:
        checkpoint_path = checkpoint_cb.best_model_path
        if os.path.isfile(checkpoint_path):
            LOGGER.info('Selected best checkpoint: %s', checkpoint_path)
            return checkpoint_path
        else:
            LOGGER.warning("Couldn't find the best checkpoint, probably callback haven't been called yet.")

    checkpoint_path = os.path.join(trainer.log_dir, 'checkpoint-from-trainer.pth')
    trainer.save_checkpoint(checkpoint_path)
    LOGGER.info('Saved checkpoint: %s.', checkpoint_path)
    return checkpoint_path
