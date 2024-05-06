import dill 
import os


__all__ = ['save_object']


def save_object(obj, file_name: str, folder_path: str = '../trained_models/transforms') -> None:
    """
    Saves an object to a specified file within a folder. If the folder doesn't exist, it's created.

    Args:
        obj: The Python object to save.
        file_name (str): The name of the file to save the object to (including extension).
        folder_path (str, optional): The path to the folder where the file should be saved. Defaults to './models'.
    """
    # Ensure folder exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Construct full path to save object
    file_path = os.path.join(folder_path, file_name)

    # Save object using dill
    with open(os.path.abspath(file_path), 'wb') as file:
        dill.dump(obj, file)

    print(f'Object saved to {os.path.abspath(file_path)}')
