import torch
from torch import nn


__all__ = ['LinearSVM']


class LinearSVM(nn.Module):
    """
    Implements a linear Support Vector Machine (SVM) model for classification tasks.

    This model can be used for multi-class classification problems. It uses a linear
    kernel to project input features into a higher dimensional space where it tries to find
    a hyperplane that best separates the classes.

    Attributes:
        input_size (int): The size of the input features.
        num_classes (int): The number of classes in the classification task.
        in_channels (int): The number of input channels. Default is 1.
        w (nn.Parameter): Weight parameter of the linear SVM.
        b (nn.Parameter): Bias parameter of the linear SVM.
    
    Methods:
        forward(x: torch.Tensor): Performs a forward pass of the SVM model.
    """

    def __init__(self, input_size: int, num_classes: int, in_channels: int = 1) -> None:
        """
        Initializes the LinearSVM model with the input size, number of classes,
        and the number of input channels.

        Args:
            input_size (int): The size of the input features.
            num_classes (int): The number of classes in the classification task.
            in_channels (int): The number of input channels. Default is 1.
        """
        super(LinearSVM, self).__init__()
        self.input_size = input_size * in_channels
        
        # Initialize weights and bias with correct shape
        self.W = nn.Parameter(torch.randn(num_classes, self.input_size), requires_grad=True)
        self.B = nn.Parameter(torch.randn(num_classes), requires_grad=True)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the SVM model using the input features.

        The method applies a linear transformation to the input features using the model's
        weights and bias. It returns the transformed features which can be used with a hinge
        loss during training.

        Args:
            x (torch.Tensor): The input features, a tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: The transformed features, a tensor of shape (batch_size, num_classes).
        """
        # Flatten input in case it's a tensor with more than 2 dimensions
        X = X.view(-1, self.input_size)
        
        # Apply linear transformation
        H = X.matmul(self.W.t()) + self.B
        return H