import torch
from torch import nn

__all__ = ['HingeLoss']

class HingeLoss(nn.Module):
    """
    A PyTorch module for computing the hinge loss, suitable for training classifiers,
    particularly with Support Vector Machine (SVM) methodology.
    
    This loss function aims to maximize the margin between decision boundaries and data points.
    It ensures the score of the correct class is higher than incorrect ones by at least a specified margin.
    """
    def __init__(self, margin: float = 1.0):
        """
        Initializes the HingeLoss module.

        Args:
            margin (float, optional): The margin parameter for the hinge loss. Defaults to 1.0.
        """
        super(HingeLoss, self).__init__()
        self.margin = margin

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculates the hinge loss for a batch of outputs and corresponding ground truth labels.

        Args:
            outputs (torch.Tensor): The predicted scores for each class. A tensor of shape
                                    (batch_size, num_classes), where each row represents scores
                                    for each class for a single sample.
            labels (torch.Tensor): The ground truth labels. A tensor of shape (batch_size,)
                                   containing the index of the correct class for each sample.

        Returns:
            torch.Tensor: The calculated hinge loss for the batch.
        """
        n = labels.size(0)  # Number of samples
        correct_scores = outputs[torch.arange(n), labels].unsqueeze(1)
        margins = outputs - correct_scores + self.margin
        loss = torch.max(margins, torch.tensor(0.0)).mean() - self.margin
        return loss