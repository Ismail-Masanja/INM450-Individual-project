from typing import Any
import torch
import os


__all__ = ['save_pytorch_model']


def save_pytorch_model(model: Any, model_name: str, folder_path: str = '../data/trained_models') -> None:
    """
    Saves a PyTorch model to a specified directory with a given name.

    Parameters:
        model (Any): The PyTorch model to be saved.
        model_name (str): The name to be used for saving the model. This name will also be used as the filename.
        folder_path (str, optional): The path to the folder where the model should be saved. Defaults to './models'.

    Returns:
        None
    """
    folder_path = os.path.abspath(folder_path)
    # Ensure folder exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Construct full path to save model
    model_path = os.path.join(folder_path, f"{model_name}.model")

    # Save model
    torch.save(model, os.path.abspath(model_path))
    print(f'Model: {model_name} saved!')
    print(f'Path: {os.path.abspath(model_path)}')
