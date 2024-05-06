import polars as pl
import tempfile
import shutil
from typing import List


__all__ = ['save_predictions']


def save_predictions(dataset_path: str, predictions: List[int], label: str) -> None:
    """
    Saves predictions to a CSV file by adding them as a new column.

    This function reads a dataset from a CSV file, adds a new column with predictions,
    and saves the modified dataset back to the same CSV file. A temporary file is used
    during the process to ensure data integrity.

    Parameters:
        dataset_path (str): The path to the CSV dataset file.
        predictions (List[int]): A list of predictions to be added as a new column.
        label (str): The name of the new column for the predictions.

    Returns:
        None
    """
    # Read dataset from the CSV file
    df = pl.read_csv(dataset_path)

    # Create a new column with predictions
    predictions_series = pl.Series(name=label, values=predictions)
    df = df.with_columns(predictions_series)

    # Write modified dataset to a temporary file
    with tempfile.NamedTemporaryFile('w', delete=False) as tmp_file:
        tmp_file_path = tmp_file.name
        df.write_csv(tmp_file_path)

    # Move temporary file to overwrite original dataset file
    shutil.move(tmp_file_path, dataset_path)
