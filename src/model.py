import torch.nn as nn
import torch.nn.functional as func
from torch import Tensor


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_ks=3, pool_ks=2, activation=func.relu):
        super().__init__()
        self.conv_ks = conv_ks
        self.activation = activation
        self.pool = nn.MaxPool2d(kernel_size=pool_ks)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=conv_ks, padding=1, stride=1)

    def forward(self, image: Tensor) -> Tensor:
        return self.pool(self.activation(self.conv(image)))


def conv_block_ks_3_pool_ks_2(in_channels, out_channels, activation=func.relu) -> nn.Module:  # noqa: WPS114
    return ConvBlock(in_channels, out_channels, conv_ks=3, pool_ks=2, activation=activation)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_net = nn.Sequential(
            conv_block_ks_3_pool_ks_2(in_channels=3, out_channels=64),
            conv_block_ks_3_pool_ks_2(in_channels=64, out_channels=128),
            conv_block_ks_3_pool_ks_2(in_channels=128, out_channels=256),
            conv_block_ks_3_pool_ks_2(in_channels=256, out_channels=512),
        )

        self.classification_head = nn.Sequential(
            nn.Linear(in_features=512 * 4 * 4, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(128),
            nn.Linear(in_features=128, out_features=6),
        )

    def forward(self, image: Tensor) -> Tensor:
        features = self.conv_net(image)
        features = features.view(features.size(0), -1)
        return self.classification_head(features)
