import torch
from torch import nn


__all__ = ['SimpleResNet1DCNN']


class ResidualBlock1D(nn.Module):
    """
    Implements a 1D Residual Block for a ResNet-like architecture.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        downsample (nn.Module, optional): Downsampling module if needed for matching dimensions. Defaults to None.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: nn.Module = None):
        super(ResidualBlock1D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels,
                      kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.PReLU(num_parameters=out_channels),
            nn.Conv1d(out_channels, out_channels,
                      kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels)
        )
        self.downsample = downsample
        self.relu = nn.PReLU(num_parameters=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.block(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class SimpleResNet1DCNN(nn.Module):
    """
    A simple 1D implementation of a ResNet architecture.

    Args:
        channels (int): Number of input channels.
        hidden_units (int): Number of hidden units in the convolutional layers.
        output_units (int): Number of output units (classes).
        sequence_length (int): Length of the input sequences.
    """

    def __init__(self, channels: int, hidden_units: int, output_units: int, sequence_length: int):
        super(SimpleResNet1DCNN, self).__init__()

        self.initial_block = nn.Sequential(
            nn.Conv1d(channels, hidden_units, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(hidden_units),
            nn.PReLU(num_parameters=hidden_units),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            ResidualBlock1D(hidden_units, hidden_units),
            ResidualBlock1D(hidden_units, hidden_units)
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.LayerNorm(hidden_units),
            nn.PReLU(num_parameters=hidden_units),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_units, output_units)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:  # if [batch_size, sequence_length]
            x = x.unsqueeze(1)  # Add channel dimension

        out = self.initial_block(x)
        out = self.residual_blocks(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
