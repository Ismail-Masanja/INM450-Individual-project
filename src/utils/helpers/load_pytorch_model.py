import torch
from typing import Any
import os


__all__ = ['load_pytorch_model']


def load_pytorch_model(model_name: str, folder_path: str = './models') -> Any:
    """
    Loads a PyTorch model from a specified directory.

    Parameters:
        model_name (str): The name of the model to load. This name should match the filename used when the model was saved.
        folder_path (str, optional): The path to the folder where the model is saved. Defaults to './models'.

    Returns:
        Any: The loaded PyTorch model.
    """
    # Construct full path to model file
    model_path = os.path.join(folder_path, f"{model_name}.model")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure model file exists
    if not os.path.exists(os.path.abspath(model_path)):
        raise FileNotFoundError(f"The model file '{model_path}' does not exist.")

    # Load and return model
    model = torch.load(os.path.abspath(model_path), map_location=torch.device(device))
    return model
