import torch.cuda
from lightning import Callback, LightningModule, Trainer
from torchinfo import summary
from torchvision.utils import make_grid

from src.lightning_module import ClassificationLightningModule


class LogModelSummary(Callback):
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        images = next(iter(trainer.train_dataloader))[0]

        images = images.to(pl_module.device)
        pl_module: ClassificationLightningModule = trainer.model
        summary(pl_module.model, input_data=images)


class VisualizeBatch(Callback):
    def __init__(self, every_n_epochs: int):
        super().__init__()
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_start(self, trainer: Trainer, pl_module: ClassificationLightningModule):  # noqa: WPS210
        if trainer.current_epoch % self.every_n_epochs == 0:
            images = next(iter(trainer.train_dataloader))[0]

            visualizations = []
            for img in images:
                visualizations.append(
                    img.mul(255).add_(0.5).clamp_(0, 255).to(pl_module.device, torch.uint8),  # noqa: WPS221, WPS432
                )
            grid = make_grid(visualizations, normalize=False)
            trainer.logger.experiment.add_image(
                'Batch preview',
                img_tensor=grid,
                global_step=trainer.global_step,
            )
