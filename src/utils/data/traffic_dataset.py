from collections import namedtuple
from typing import Optional, Callable, Dict
from torch.utils.data import Dataset
import torch
import polars as pl


__all__ = ['TrafficDataset']


class TrafficDataset(Dataset):
    """ A custom dataset for loading network traffic data from a CSV file.

    Attributes:
        csv_file (str): Path to the CSV file containing the dataset.
        label (str): Column name in the CSV file to use as the label.
        drop (Optional[list of str]): Columns to be excluded from the dataset.
        transform (Optional[Callable]): A function/transform that takes in
            a sample and returns a transformed version.
        label_map (Optional[Dict[str, int]]): A dictionary mapping label names
            to integers for classification.

    The dataset reads a CSV file, optionally drops specified columns,
    and can apply a transform to the data.
    """

    def __init__(self, csv_file: str, label: str, drop: Optional[list] = None,
                 transform: Optional[Callable] = None,
                 label_map: Optional[Dict[str, int]] = None):
        """Initialize the dataset with CSV file, label column, and optionally 
        columns to drop, a transform, and a label map."""

        self.transform = transform

        # Modify all keys to lower case to avoid duplicates
        self.label_map = label_map
        if self.label_map is not None:
            self.label_map = {k.lower(): v for k, v in label_map.items()}

        # Load the dataset using Polars
        self.dataframe = pl.read_csv(csv_file)

        # Drop columns
        self.dataframe = self.dataframe.drop(drop)

        # Define a named tuple with field names matching dataset columns, excluding dropped, Label
        self.Traffic = namedtuple(
            'Traffic', [col for col in self.dataframe.columns if col != label])

        # Separate features and labels
        self.features = self.dataframe.drop(label)
        self.labels = self.dataframe.select(label)

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return self.dataframe.height

    def __getitem__(self, idx: int) -> tuple:
        """Retrieves the ith sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the features as a namedtuple and the label 
            as a tensor.
        """

        traffic_row = self.features.row(idx)
        traffic = self.Traffic._make(traffic_row)
        label = self.labels.row(idx)

        if self.transform is not None:
            traffic = self.transform(traffic)

        if self.label_map is not None:
            label = torch.tensor(self.label_map.get(str(*label).lower()))
        else:
            label = torch.tensor(int(label[0]))

        return traffic, label
