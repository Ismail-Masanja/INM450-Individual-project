#! /bin/bash

SCRIPT='./src/main.py' # path to main function
CONFIG_FILE='./config.ini' # path to configuration file.

echo 'Starting Program Check Vet Directory to Vet IPs to Block'

# run the main python script in the background (nohup) in failure exit
sudo nohup python $SCRIPT $CONFIG_FILE 2> /dev/null || { echo 'Python script failed' ; exit 1; }