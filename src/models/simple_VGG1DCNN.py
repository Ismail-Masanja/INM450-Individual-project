import torch
from torch import nn


__all__ = ['SimpleVGG1DCNN']


class SimpleVGG1DCNN(nn.Module):
    """
    A simple 1D VGG-like Convolutional Neural Network for sequence data.

    Args:
        channels (int): Number of input channels.
        hidden_units (int): Number of hidden units in the convolutional layers.
        output_units (int): Number of output units (classes).
        sequence_length (int): Length of the input sequences.

    Attributes:
        input_block (nn.Sequential): The input convolutional block.
        hidden_block (nn.Sequential): The hidden convolutional block.
        classifier (nn.Sequential): The classifier block.
        num_flat_features (int): The number of features to be flattened before passing to the classifier.
    """

    def __init__(self, channels: int, hidden_units: int, output_units: int, sequence_length: int):
        super().__init__()

        # Input convolutional block
        self.input_block = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=hidden_units,
                      kernel_size=4, stride=1, padding=1),
            nn.BatchNorm1d(hidden_units),
            nn.PReLU(num_parameters=hidden_units),
            nn.Conv1d(in_channels=hidden_units, out_channels=hidden_units,
                      kernel_size=4, stride=1, padding=1),
            nn.BatchNorm1d(hidden_units),
            nn.PReLU(num_parameters=hidden_units),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Hidden convolutional block
        self.hidden_block = nn.Sequential(
            nn.Conv1d(in_channels=hidden_units, out_channels=hidden_units,
                      kernel_size=4, stride=1, padding=1),
            nn.BatchNorm1d(hidden_units),
            nn.PReLU(num_parameters=hidden_units),
            nn.Conv1d(in_channels=hidden_units, out_channels=hidden_units,
                      kernel_size=4, stride=1, padding=1),
            nn.BatchNorm1d(hidden_units),
            nn.PReLU(num_parameters=hidden_units),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Calculate flattening size
        self.num_flat_features = self._get_conv_output_size(sequence_length, channels)

        # Classifier block
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.num_flat_features,
                      out_features=self.num_flat_features),
            nn.LayerNorm(self.num_flat_features),
            nn.PReLU(num_parameters=self.num_flat_features),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=self.num_flat_features, out_features=output_units)
        )

    def _get_conv_output_size(self, sequence_length: int, channels: int) -> int:
        """
        Determines the size of the tensor after passing through convolutional blocks.

        Args:
            sequence_length (int): Length of the input sequences.
            channels (int): Number of input channels.

        Returns:
            int: The total number of features after convolutions and pooling.
        """
        dummy_input = torch.zeros(1, channels, sequence_length)
        output = self.input_block(dummy_input)
        output = self.hidden_block(output)
        return int(torch.prod(torch.tensor(output.size()[1:])))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            input (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the model.
        """
        if input.dim() == 2:  # if [batch_size, sequence_length]
            input = input.unsqueeze(1)  # Add channel dimension
        _output1 = self.input_block(input)
        _output2 = self.hidden_block(_output1)
        output = self.classifier(_output2)
        return output
