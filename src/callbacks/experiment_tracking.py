from typing import Dict, List

import torch
from clearml import Task
from lightning import Callback, LightningModule, Trainer
from sklearn.metrics import confusion_matrix
from torch import Tensor


class LogConfusionMatrix(Callback):
    def __init__(self, task: Task, idx_to_class: Dict[int, str]):
        super().__init__()
        self.task = task
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
        self.task.logger.report_confusion_matrix(
            'Confusion matrix',
            'ignored',
            iteration=trainer.global_step,
            matrix=cf_matrix,
            xaxis=None,
            yaxis=None,
            xlabels=cf_labels,
            ylabels=cf_labels,
        )
