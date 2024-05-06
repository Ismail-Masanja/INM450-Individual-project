import csv
import os
from datetime import datetime
from typing import List, Dict, Any


__all__ = ['save_vet_data']


def save_vet_data(vet_data: List[Dict[str, Any]], directory: str) -> None:
    """
    Saves or updates network vetting data in a CSV file located in the specified directory.

    This function checks if the 'vet_data.csv' file exists in the provided directory. If it exists, 
    it loads the existing data and updates it with the new entries or modifications from `vet_data`. 
    If the file does not exist, it creates a new file and writes the data from `vet_data` to it. 
    After updating or writing the data, it sets the file's permissions to be read and write for everyone.

    Args:
        vet_data (List[Dict[str, Any]]): A list of dictionaries where each dictionary contains details 
                                         about a network connection that needs to be vetted. Expected 
                                         keys in each dictionary include 'connection_ip', 'from_ports', 
                                         'to_ports', and 'suspected_attacks'.
        directory (str): The path to the directory where the 'vet_data.csv' file will be stored or updated.

    Side Effects:
        - Reads from or writes to a file named 'vet_data.csv' located in the specified directory.
        - Changes the file system permissions of 'vet_data.csv' to allow read and write operations 
          for all users (chmod 666).

    Returns:
        None: This function does not return any value.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
        PermissionError: If the script lacks permissions to read from or write to the directory.
    """

    # Define file path
    file_path = os.path.join(directory, 'vet_data.csv')

    # Check if file exists
    if os.path.exists(file_path):
        # Load existing data
        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            existing_data = list(reader)
    else:
        existing_data = []

    # Convert existing data into a dict of dicts for easier manipulation
    existing_dict = {item['connection_ip']: item for item in existing_data}

    # Update existing data or add new entries
    for entry in vet_data:
        ip = entry['connection_ip']
        if ip in existing_dict:
            # Update time
            existing_dict[ip]['time'] = datetime.now().isoformat()
            # Update ports and suspected attacks
            existing_dict[ip]['from_ports'] = update_set(
                existing_dict[ip].get('from_ports', set()), entry['from_ports'])
            existing_dict[ip]['to_ports'] = update_set(
                existing_dict[ip].get('to_ports', set()), entry['to_ports'])
            existing_dict[ip]['suspected_attacks'] = update_set(
                existing_dict[ip].get('suspected_attacks', set()), entry['suspected_attacks'])
        else:
            # Add new entry
            new_entry = {
                'time': datetime.now().isoformat(),
                'Action': 'unblock',
                'connection_ip': ip,
                'from_ports': set_to_string(entry['from_ports']),
                'to_ports': set_to_string(entry['to_ports']),
                'suspected_attacks': set_to_string(entry['suspected_attacks']),
            }
            existing_dict[ip] = new_entry

    # Save updated data to CSV
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        fieldnames = ['time', 'Action', 'connection_ip', 'from_ports',
                      'to_ports', 'suspected_attacks', ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in existing_dict.values():
            writer.writerow(item)

    # Set file permissions to read and write for everyone
    os.chmod(file_path, 0o666)  # Permissions set to rw-rw-rw-


def update_set(existing, new):
    updated_set = set(eval(existing)) if existing else set()
    updated_set.update(new)
    return set_to_string(updated_set)


def set_to_string(s):
    return str(s)
