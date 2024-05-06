import dill 
import os

__all__ = ['load_object']


def load_object(file_name: str, folder_path: str = '../trained_models/transforms') -> any:
    """
    Loads a Python object from a specified file within a folder.

    Args:
        file_name (str): The name of the file to load the object from (including extension).
        folder_path (str, optional): The path to the folder where the file is saved. Defaults to './models'.

    Returns:
        The Python object loaded from the file.
    """
    # Construct full path to file
    file_path = os.path.join(folder_path, file_name)

    # Load and return object using dill 
    if os.path.exists(os.path.abspath(file_path)):
        with open(os.path.abspath(file_path), 'rb') as file:
            return dill.load(file)
    else:
        raise FileNotFoundError(f"File not found: {os.path.abspath(file_path)}")
