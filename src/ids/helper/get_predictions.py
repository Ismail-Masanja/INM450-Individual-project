import torch
from torch.utils.data import DataLoader
from typing import List


__all__ = ['get_predictions']


def get_predictions(model: torch.nn.Module, dataloader: DataLoader) -> List[int]:
    """
    Obtain predictions from a model given data from a DataLoader.

    This function sets the model to evaluation mode, iterates over the given DataLoader,
    and computes predictions by selecting the class with the highest output score for each input.
    Predictions are aggregated and returned as a list of integers representing the predicted classes.

    Parameters:
        model (torch.nn.Module): The trained model from which to get predictions.
        dataloader (DataLoader): The DataLoader providing batches of data for prediction.

    Returns:
        List[int]: A list of predicted class indices.
    """
    model.eval()  # Ensure the model is in evaluation mode.
    predictions = []  # Initialize an empty list to store predictions.

    with torch.no_grad():  # Temporarily set all requires_grad flags to false.
        for features, _ in dataloader:
            # Compute the model output
            output = model(features)
            # Get the index of the max log-probability as prediction
            pred = output.argmax(dim=1)
            # Extend  predictions list with current batch predictions
            predictions.extend(pred.cpu().tolist())

    return predictions
