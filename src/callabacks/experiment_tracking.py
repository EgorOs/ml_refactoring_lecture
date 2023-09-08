from typing import Optional, List, Dict

import torch
from clearml import Task
from lightning import Callback, Trainer, LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from sklearn.metrics import confusion_matrix
from torch import Tensor


class EvaluationResults(Callback):
    def __init__(self, task: Task, class_to_idx: Dict[str, int]):
        super().__init__()
        self.task = task
        self.class_to_idx = class_to_idx
        self.preds = []
        self.targets = []

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: List[Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.preds.append(outputs)
        self.targets.append(batch[1])

    @property
    def idx_to_class(self) -> Dict[int, str]:
        return {idx: cl for cl, idx in self.class_to_idx.items()}

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        cf_matrix = confusion_matrix(
            torch.cat(self.targets, dim=0).cpu().numpy(),
            torch.cat(self.preds, dim=0).cpu().numpy(),
        )
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
