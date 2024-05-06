import torch


__all__ = ['StandardScalerTransform']


class StandardScalerTransform:
    """Transform class for applying standard scaling to tensor data.

    This class performs standard scaling on input data by subtracting the mean and dividing
    by the standard deviation. The mean and standard deviation used for scaling are provided
    during initialization. This normalization technique is commonly used in data preprocessing
    to standardize the range of features for machine learning models.

    Attributes:
        mean (torch.Tensor): The mean value(s) used for scaling. Can be a scalar or a tensor
                             with dimensions matching those of the input data to be scaled.
        std (torch.Tensor): The standard deviation value(s) used for scaling. Can be a scalar
                            or a tensor with dimensions matching those of the input data to
                            be scaled.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        """Initializes the scaler with pre-calculated mean and standard deviation.

        Args:
            mean (torch.Tensor): The mean value(s) for scaling.
            std (torch.Tensor): The standard deviation value(s) for scaling.
        """
        self.mean = mean
        self.std = std

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Applies standard scaling to the input data.

        Args:
            data (torch.Tensor): The input data tensor to be normalized.

        Returns:
            torch.Tensor: The normalized data tensor.
        """
        # Add a small epsilon to avoid division by zero in case std is zero
        epsilon = 1e-6
        normalized_data = (data - self.mean) / (self.std + epsilon)
        return normalized_data
