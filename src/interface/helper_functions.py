import os
import csv
import sys
import threading
from datetime import datetime

from ids.utils import block_ip

__all__ = ['find_csv', 'process_csv', 'setup_logging',
           'manage_log_size', 'start_thread', 'time_str']


def find_csv(directory):
    """ Find the first CSV file in the specified directory. """
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            return os.path.join(directory, filename)
    return None


def setup_logging(log_file):
    sys.stdout = open(log_file, 'a')
    sys.stderr = open(log_file, 'a')


def manage_log_size(log_file, max_size_mb=5):
    """ Prevents the log size from growing too large """
    max_size = max_size_mb * 1024 * 1024  # Convert MB to bytes
    # Check the current size of the file
    if os.path.getsize(log_file) > max_size:
        # Open the current log, read contents
        with open(log_file, 'r+') as file:
            content = file.readlines()
            # Calculate the midpoint
            midpoint = len(content) // 2
            # Keep only the second half of the content
            file.seek(0)
            file.writelines(content[midpoint:])
            file.truncate()


def start_thread(target, interval):
    """ Helper function to start a thread for a given task function. """
    thread = threading.Thread(target=target, args=(interval,))
    thread.daemon = True  # Allows program to exit even if thread is running
    thread.start()
    return thread


def process_csv(file_path):
    """ Process the CSV file to apply block_ip function to each IP and action. """
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            ip = row['connection_ip']
            action = str(row['Action']).lower().strip()
            block_ip.block_ip(ip, action)


def time_str():
    current_time = datetime.now()
    return current_time.strftime("%Y-%m-%d %H:%M:%S")
