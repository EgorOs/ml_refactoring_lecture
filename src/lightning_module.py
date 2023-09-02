from typing import Dict, List

import torch
import torch.nn.functional as func
from lightning import LightningModule
from torch import Tensor
from torchmetrics import MeanMetric

from src.model import CNN


class ClassificationLightningModule(LightningModule):
    def __init__(self):
        super().__init__()
        self._train_loss = MeanMetric()

        self.model = CNN()

        self.save_hyperparameters()

    def forward(self, images: Tensor) -> Tensor:
        return self.model(images)

    def training_step(self, batch: List[Tensor]) -> Dict[str, Tensor]:  # noqa: WPS210
        images, targets = batch
        logits = self(images)
        probs = func.softmax(logits, dim=1)
        loss = func.cross_entropy(probs, targets)
        self._train_loss(loss)
        self.log('step_loss', loss, on_step=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def on_train_epoch_end(self) -> None:
        self.log('mean_train_loss', self._train_loss.compute(), on_step=False, prog_bar=True, on_epoch=True)
        self._train_loss.reset()

    def validation_step(self, batch: List[Tensor], batch_idx: int) -> None:
        pass

    def test_step(self, batch: List[Tensor], batch_idx: int) -> None:
        pass

    def configure_optimizers(self) -> dict:
        # TODO: parametrize optimizer and lr scheduler.
        optimizer = torch.optim.SGD(self.parameters(), lr=2e-3)  # noqa: WPS432 will be parametrized
        # scheduler = get_cosine_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=70,  # noqa: WPS432 will be parametrized
        #     num_training_steps=self.trainer.estimated_stepping_batches,
        #     num_cycles=0.4,  # noqa: WPS432 will be parametrized
        # )
        return {
            'optimizer': optimizer,
            # 'lr_scheduler': {
            #     'scheduler': scheduler,
            #     'interval': 'step',
            #     'frequency': 1,
            # },
        }
