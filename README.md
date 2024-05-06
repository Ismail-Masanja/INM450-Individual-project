# Introduction

This project aims to use machine / deep learning models to detect, classify and defend against malicious network activity. There are three subsections:

1. Setting up the project
2. Running the Program
3. Training machine / deep learning models ( optional ) - Best performing models are already saved in `data/trained_models`. If you wish to train more models you can proceed with this section.

## 1. Project Setup

1.1. Ensure **python** and **pip** is installed. Navigate to this folder in terminal and type.

```bash
  python --version 
  pip --version
```

If these commands show python and pip versions we can proceed if not, instructions to install python can be found [here](https://wiki.python.org/moin/BeginnersGuide/Download), and for installing pip [here](https://pip.pypa.io/en/stable/installation/).

1.2. It is recommended to setup a new  python  virtual enviroment. This keeps project's dependacies separate from the global python enviroment and other projects. To activate:

- On windows

```bash
python3.9 -m venv myenv
myenv\Scripts\activate
```

- On Unix/Linux systems (MacOS and Linux)

```bash
python3.9 -m venv myenv
source myenv/bin/activate
```

In case of any errors, these are likely because python3.9 is not installed. Instructions on installing python3.9 can be found [here](https://gist.github.com/MichaelCurrin/3a4d14ba1763b4d6a1884f56a01412b7).

1.3. Make sure in your terminal, you have navigated to this project directory where the `requirements.txt` is and run:

```bash
pip install -r requirements.txt
```

In case of any errors, make sure downgrade your python version to 3.9. Instructions can be found [here](https://gist.github.com/MichaelCurrin/3a4d14ba1763b4d6a1884f56a01412b7). Repeat 1.1 to 1.3 after.

## 2. Run the Program

To run the program all we need to do is run the script `run.sh` before that, we need to set it to executable status and modify `config.ini` for specific required configurations.

2.1. Ensure the bash script `run.sh` is executable. In your terminal the current working directory should be this project where `run.sh` is located run:

``` bash
chmod +x run.sh
```

**Note:** For windows systems, for these terminal commands and ones that follow, it is recommended to use Windows subsystem for linux. More details on how to install and use windows subsystem for linux (WSL) can be found [here](https://learn.microsoft.com/en-us/windows/wsl/install). If this fails, to run the program, navigate to `src` and run `python main.py <config file location>`.

2.2 Setting up `config.ini`. This is where program configurations are. There is a total of 9 different configurations to be made. The full configuration file is as follows:

```ini
[DEFAULT]
INTERFACE = en0
CAPTURE_DURATION = 5
INTERVAL = 60
VET_UPDATE_INTERVAL = 15
detection_model = rf_model_detection
classification_model = xgb_model_classification
vet_dir = <Absolute path to the Vet directory> 
data_dir = <Absolute path to the data directory>
log_file = <Absolute path to the log directory>
```

- **INTERFACE**:
  - `en0`
  - This specifies the network interface that the application will monitor or use. For windows systems instructions on how to find network interfaces can be found [here](https://www.computerhope.com/issues/ch000500.htm). For Unix and Linux systems [here](https://www.cyberciti.biz/faq/linux-list-network-interfaces-names-command/).

- **CAPTURE_DURATION**:
  - `5` (in seconds)
  - Defines the duration for which network traffica data from the interface is captured continuously during a session. Reccommended values are `3` for short bursty aggressive scans, `5` for standard mode.

- **INTERVAL**:
  - `60` (in seconds)
  - The time interval between consecutive network traffic data processing and monitoring cycles.

- **VET_UPDATE_INTERVAL**:
  - `15` (in seconds)
  - Interval at which the vetting process / file is refreshed or checked.

- **detection_model**:
  - `rf_model_detection`
  - Specifies the path or identifier for the machine learning model used for detection tasks.

- **classification_model**:
  - `xgb_model_classification`
  - Specifies the path or identifier for the machine learning model used for classification tasks.

- **vet_dir**:
  - `<Absolute path to the Vet directory>`
  - Directory path where vetting csv file is stored. If not available in the path a new directory is created.

- **data_dir**:
  - `<Absolute path to the data directory>`
  - Directory path used to store application data, which  include `network_traffic` captured, `trained_models` and `training_data`.  If not available in the path an error is thrown.

- **log_file**:
  - `<Absolute path to the log directory>`
  - Path to the file where application logs are written. This file is used for debugging and tracking application behavior. If not available in the path a new directory is created.

2.3. Run the application. To run the application, run the command below:

```bash
sudo ./run.sh
```

Note: Make sure you are in the same directory as `run.sh` script in the project main directory. `sudo` is required because capturing network interface traffic require root priviledges.

2.4. To see possible malicious connections, navigate to `vet` directory, open `vet_data.csv`. Further investigation to ascertain that the connection is indeed malicious. Instructions that can help in further vetting can be found [here](https://tip.kaspersky.com/Help/Doc_data/en-US/IpInvestigation.htm). Once a malicious connecting is determined, change `unblock` to `block` in the respective row and save the file. `block` -ed connections will be blocked.

## 3. Train Machine / Deep Learning models (Optional)

3.1. Download original corpus data. To Get training data download the Corpus data from: <https://rdm.uq.edu.au/files/e2412450-ef9c-11ed-827d-e762de186848>. Move the downloaded file into `data` directory.

3.2. In `Src/scripts` Run following commands to make sure scripts are executable:

```bash
chmod a+x preprocess_data.sh prepare_detection_data.sh prepare_classification_data.sh
```

3.3. Run preprocess script.

```bash
./preprocess_data.sh
```

3.4. Run prepare_detection to extract detection data
Usage ./prepare_detection_data.sh \<Total Number of Samples> \<Train Ratio> \<Val Ratio> \<Test Ratio>
Example:

```bash
./prepare_detection_data.sh 20000 0.8 0.1 0.1
```

Same for ./prepare_classification_data.sh
Usage ./prepare_classification_data.sh \<Total Number of Samples> \<Train Ratio> \<Val Ratio> \<Test Ratio>
Example:

```bash
./prepare_classification_data.sh 20000 0.8 0.1 0.1
```

Two folders will be created, `classification_data` and `detection_data` move this folders into `data/training_data`.

3.5. Run the jupyter notebooks in `notebooks` folder. Models will be saved into `data/trained_models`.
