from typing import Tuple
import torch
from torch.utils.data import DataLoader


__all__ = ['dataset_mean_std']


def dataset_mean_std(dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the mean and standard deviation of a dataset represented by a DataLoader.

    This function iterates through the DataLoader, accumulating the sum and sum of squares
    of the dataset elements to compute the mean and standard deviation. It ensures numerical
    stability by replacing zero mean values with 1 and non-positive standard deviations with 0.5.

    Args:
        dataloader (DataLoader): A DataLoader object representing the dataset for which
                                 mean and standard deviation are to be calculated.

    Returns:
        torch.Tensor: The mean values of the dataset across each dimension.
        torch.Tensor: The standard deviation values of the dataset across each dimension.

    Note:
        This function assumes that the DataLoader yields batches of data where each batch
        is a tensor with a consistent shape, dtype, and device location.
    """

    # Peek the first batch to get the shape, dtype, and device
    first_batch_data, _ = next(iter(dataloader))
    device = first_batch_data.device
    dtype = first_batch_data.dtype

    # Initialize sum and sum_squared tensors
    sum_ = torch.zeros(first_batch_data[0].size(), dtype=dtype, device=device)
    sum_squared = torch.zeros_like(sum_)

    sample_count = 0

    # Loop through the DataLoader and accumulate sums
    for data, _ in dataloader:
        sum_ += torch.sum(data, dim=0)
        sum_squared += torch.sum(data ** 2, dim=0)
        sample_count += data.size(0)

    # Calculate mean and variance
    mean = sum_ / sample_count
    variance = (sum_squared / sample_count) - (mean ** 2)
    # Calculate standard deviation and ensure it's non-negative
    std_dev = torch.sqrt(variance.clamp(min=0))

    # Replace zero mean values with 1 and non-positive standard deviations with 0.5
    mean = torch.where(mean == 0, torch.tensor(1.0, dtype=dtype, device=device), mean)
    std_dev = torch.where(std_dev <= 0, torch.tensor(
        0.5, dtype=dtype, device=device), std_dev)

    return mean, std_dev
