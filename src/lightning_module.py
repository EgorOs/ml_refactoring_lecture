from typing import Dict, List

import torch
import torch.nn.functional as func
from lightning import LightningModule
from torch import Tensor
from torchmetrics import MeanMetric

from src.metrics import get_metrics
from src.model import CNN
from src.schedulers import get_cosine_schedule_with_warmup


class ClassificationLightningModule(LightningModule):  # noqa: WPS214
    def __init__(self, class_to_idx: Dict[str, int]):
        super().__init__()
        self._train_loss = MeanMetric()
        self._valid_loss = MeanMetric()

        metrics = get_metrics(
            num_classes=len(class_to_idx),
            num_labels=len(class_to_idx),
            task='multiclass',
            average='macro',
        )
        self._valid_metrics = metrics.clone(prefix='valid_')
        self._test_metrics = metrics.clone(prefix='test_')

        self.model = CNN()

        self.save_hyperparameters()

    def forward(self, images: Tensor) -> Tensor:
        return self.model(images)

    def training_step(self, batch: List[Tensor]) -> Dict[str, Tensor]:  # noqa: WPS210
        images, targets = batch
        logits = self(images)
        loss = func.cross_entropy(logits, targets)
        self._train_loss(loss)
        self.log('step_loss', loss, on_step=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def on_train_epoch_end(self) -> None:
        self.log('mean_train_loss', self._train_loss.compute(), on_step=False, prog_bar=True, on_epoch=True)
        self._train_loss.reset()

    def validation_step(self, batch: List[Tensor], batch_idx: int) -> None:
        images, targets = batch
        logits = self(images)
        self._valid_loss(func.cross_entropy(logits, targets))

        self._valid_metrics(logits, targets)

    def on_validation_epoch_end(self) -> None:
        self.log('mean_valid_loss', self._valid_loss.compute(), on_step=False, prog_bar=True, on_epoch=True)
        self._valid_loss.reset()

        self.log_dict(self._valid_metrics.compute(), prog_bar=True, on_epoch=True)
        self._valid_metrics.reset()

    def test_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        images, targets = batch
        logits = self(images)

        preds = torch.argmax(logits, dim=1)
        self._test_metrics(logits, targets)
        return preds

    def on_test_epoch_end(self) -> None:
        self.log_dict(self._test_metrics.compute(), prog_bar=True, on_epoch=True)
        self._test_metrics.reset()

    def configure_optimizers(self) -> dict:
        # TODO: parametrize optimizer and lr scheduler.
        optimizer = torch.optim.SGD(self.model.parameters(), lr=2e-3)  # noqa: WPS432 will be parametrized
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=200,  # noqa: WPS432 will be parametrized
            num_training_steps=self.trainer.estimated_stepping_batches,
            num_cycles=0.4,  # noqa: WPS432 will be parametrized
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            },
        }
