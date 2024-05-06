import os
import time
import argparse
import configparser
from ids import predict_traffic
from interface import save_vet_data, process_csv, find_csv, start_thread, setup_logging, manage_log_size, time_str


def load_config(config_location):
    config = configparser.ConfigParser()
    config.read(config_location)
    return config['DEFAULT']


# Parse configurations from config.ini
parser = argparse.ArgumentParser(description="Load a configuration file.")
parser.add_argument("config", help="The location of the configuration file.")
args = parser.parse_args()

config = load_config(args.config)

# create config variables
INTERFACE = config['INTERFACE']
CAPTURE_DURATION = int(config['CAPTURE_DURATION'])
INTERVAL = int(config['INTERVAL'])
VET_UPDATE_INTERVAL = int(config['VET_UPDATE_INTERVAL'])
detection_model = config['detection_model']
classification_model = config['classification_model']
vet_dir = config['vet_dir']
data_dir = config['data_dir']
log_file = config['log_file']


data_dir = os.path.join(data_dir)
if not os.path.exists(data_dir):
    raise FileNotFoundError(
        'The Data directory not found!, \nCannot find trained models, \nCannot find Network Capture data')

vet_dir = os.path.join(vet_dir)
if not os.path.exists(vet_dir):
    os.makedirs(vet_dir)

log_file = os.path.join(log_file)
log_dir = os.path.dirname(log_file)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Creates the capture and predict loop
def predict(interval):
    setup_logging(log_file)
    while True:
        manage_log_size(log_file)
        save_vet_data(predict_traffic(data_dir, INTERFACE, CAPTURE_DURATION,
                                      detection_model, classification_model), vet_dir)
        time.sleep(interval)

# Creates a loop to check vet dir and block IPs
def defend(interval):
    setup_logging(log_file)
    while True:
        manage_log_size(log_file)

        csv_file = find_csv(vet_dir)
        if csv_file:
            process_csv(csv_file)
        else:
            print(f'{time_str()}: No CSV file found in the directory.')

        time.sleep(interval)


def main():
    th1 = start_thread(predict, INTERVAL)

    th2 = start_thread(defend, VET_UPDATE_INTERVAL)

    try:
        while True:
            time.sleep(5)  # Sleep main thread to prevent high CPU usage
    except KeyboardInterrupt:
        print(f'{time_str()}: Stopped by user.')


if __name__ == '__main__':
    main()
