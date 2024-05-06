from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple


__all__ = ['collect_dataloader_data']


def collect_dataloader_data(dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collects and aggregates features and labels from a DataLoader into NumPy arrays.

    Args:
        dataloader (DataLoader): The DataLoader instance from which to collect the data.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays; the first for features and the second for labels.
    """
    features_list = []
    labels_list = []

    # Iterate over DataLoader accumulate features, labels
    for features, labels in dataloader:
        # Assuming features and labels tensors, convert NumPy arrays
        features_list.append(features.numpy())
        labels_list.append(labels.numpy())

    # Concatenate list of arrays into a single NumPy array (features, labels)
    features_array = np.concatenate(features_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)

    return features_array, labels_array
