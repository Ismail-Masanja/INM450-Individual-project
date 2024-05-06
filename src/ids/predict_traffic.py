import os
from ids.utils import detect_classify
from network_capture import capture_network_traffic
from utils.helpers import load_pytorch_model
from utils.helpers import load_object
from ids.helper import transform_network_data
from typing import Any, Dict


__all__ = ['predict_traffic']


def predict_traffic(data_dir: str, INTERFACE: str, CAPTURE_DURATION: int,
                    detection_model: str, classification_model: str) -> Dict[str, Any]:
    """
    Captures network traffic, processes it through detection and classification models, 
    and returns analyzed network traffic details.

    Args:
        data_dir (str): Directory path where the network data and models are stored.
        INTERFACE (str): Network interface from which to capture the traffic.
        CAPTURE_DURATION (int): Duration in seconds for which to capture the traffic.
        detection_model (str): Filename of the pre-trained PyTorch model for traffic detection.
        classification_model (str): Filename of the pre-trained PyTorch model for traffic classification.

    Returns:
        Dict[str, Any]: A dictionary containing details of the analyzed network traffic. 
                        The structure and contents of the dictionary depend on the 
                        output of the `transform_network_data` function.
    """
    
    data_net_dir = os.path.join(data_dir, 'network_traffic')
    if not os.path.exists(data_net_dir):
        os.makedirs(data_net_dir)

    dataset_dir = os.path.join(data_net_dir, 'network_traffic_analysis.csv')

    # Capture network traffic from interface.
    flow_data = capture_network_traffic(INTERFACE, CAPTURE_DURATION)
    flow_data.write_csv(os.path.abspath(dataset_dir))
    os.chmod(dataset_dir,  0o440)  # Sets the permissions to r--r-----

    # Load Models.
    detect_model = load_pytorch_model(os.path.join(
        data_dir, 'trained_models', detection_model))
    classification_model = load_pytorch_model(os.path.join(
        data_dir, 'trained_models', classification_model))

    detection_preprocessing_transforms = load_object(os.path.join(
        data_dir, 'trained_models', 'detection_preprocessing_transforms'))
    classification_preprocessing_transforms = load_object(os.path.join(
        data_dir, 'trained_models', 'classification_preprocessing_transforms'))

    # Attacks present for classification
    labels = [
        "Benign", "DDoS", "Reconnaissance", "Injection", "DoS",
        "Brute Force", "Password", "XSS", "Infilteration",
        "Exploits", "Scanning", "Fuzzers", "Backdoor", "Bot",
        "Generic", "Analysis", "Theft", "Shellcode", "MITM",
        "Worms", "Ransomware"
    ]

    # Create a dictionary with the mapping of labels
    label_maps = {label: i for i, label in enumerate(labels)}

    batch_size = 64
    drop = []

    attack_details = detect_classify(detect_model, classification_model, dataset_dir,
                                     detection_preprocessing_transforms,
                                     classification_preprocessing_transforms, drop, label_maps, batch_size)

    details = transform_network_data(attack_details, INTERFACE)

    return details
