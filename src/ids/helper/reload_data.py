from collections import namedtuple
from torch.utils.data import DataLoader
from typing import NamedTuple

from utils.data import TrafficDataset


__all__ = ['reload_data']


class ReloadedData(NamedTuple):
    """
    A NamedTuple to hold datasets and dataloaders.

    Attributes:
        pure_live_dataset: An instance of a Dataset without transformations.
        pure_live_dataloader: DataLoader for the pure_live_dataset.
        live_dataset: An instance of a Dataset with transformations applied.
        live_dataloader: DataLoader for the live_dataset.
    """
    pure_live_dataset: TrafficDataset
    pure_live_dataloader: DataLoader
    live_dataset: TrafficDataset
    live_dataloader: DataLoader


def reload_data(dataset_path: str, drop: list, transform: callable, batch_size: int) -> ReloadedData:
    """
    Reload data from a CSV file into datasets and dataloaders, optionally applying transformations.

    Parameters:
        dataset_path (str): Path to the CSV file containing the dataset.
        drop (list): List of column names to be dropped from the dataset.
        transform (callable): A function or a transform to be applied to the dataset.
        batch_size (int): Number of samples per batch to load.

    Returns:
        ReloadedData: A namedtuple containing loaded datasets and dataloaders.
    """

    label = 'L7_PROTO'
    # Load dataset
    pure_live_dataset = TrafficDataset(dataset_path, drop=drop, label=label)
    pure_live_dataloader = DataLoader(pure_live_dataset, batch_size=batch_size)

    live_dataset = TrafficDataset(dataset_path, drop=drop,
                                  label=label, transform=transform)
    live_dataloader = DataLoader(live_dataset, batch_size=batch_size, shuffle=False)

    data = namedtuple(
        'Data', 'pure_live_dataset pure_live_dataloader live_dataset live_dataloader')

    return data(pure_live_dataset, pure_live_dataloader, live_dataset, live_dataloader)
