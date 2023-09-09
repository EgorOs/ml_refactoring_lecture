from pathlib import Path
from typing import Dict, Optional

import torch
from clearml import Dataset as ClearmlDataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.config import DataConfig
from src.dataset import ClassificationDataset
from src.transform import get_train_transforms, get_valid_transforms


class ClassificationDataModule(LightningDataModule):  # noqa: WPS214
    def __init__(
        self,
        cfg: DataConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self._train_transforms = get_train_transforms(*cfg.img_size)
        self._valid_transforms = get_valid_transforms(*cfg.img_size)

        # Prevent hyperparameters from being stored in checkpoints.
        self.save_hyperparameters(logger=False)

        self.data_path: Optional[Path] = None
        self.initialized = False

        self.data_train: Optional[ClassificationDataset] = None
        self.data_val: Optional[ClassificationDataset] = None
        self.data_test: Optional[ClassificationDataset] = None

    @property
    def class_to_idx(self) -> Dict[str, int]:
        if not self.initialized:
            self.prepare_data()
            self.setup('test')
        return self.data_test.class_to_idx

    @property
    def idx_to_class(self) -> Dict[int, str]:
        return {idx: cl for cl, idx in self.class_to_idx.items()}

    def prepare_data(self):
        self.data_path = (
            Path(ClearmlDataset.get(dataset_name=self.cfg.dataset_name).get_local_copy()) / 'Classification_data'
        )

    def setup(self, stage: str):
        if stage == 'fit':
            all_data = ClassificationDataset(str(self.data_path / 'train'), transform=self._train_transforms)
            train_split = int(len(all_data) * self.cfg.data_split[0])
            val_split = len(all_data) - train_split
            self.data_train, self.data_val = torch.utils.data.random_split(  # noqa: WPS414
                all_data,
                [train_split, val_split],
            )
        elif stage == 'test':
            self.data_test = ClassificationDataset(str(self.data_path / 'test'), transform=self._valid_transforms)
        self.initialized = True

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=False,
        )
