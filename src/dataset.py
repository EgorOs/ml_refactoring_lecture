from typing import Tuple

import cv2
import torch
from numpy.typing import NDArray
from torchvision.datasets import ImageFolder


class ClassificationDataset(ImageFolder):
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img, target = self.get_raw_data(index)
        if self.transform is None:
            raise ValueError('At least np.ndarray to torch.Tensor transformation should be defined.')
        img = self.transform(image=img)['image']
        return img, target

    def get_raw_data(self, index: int) -> Tuple[NDArray[int], int]:
        path, target = self.samples[index]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, target
