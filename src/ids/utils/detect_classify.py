import torch
from typing import List, Dict, Any 
from ids.helper import reload_data, get_predictions, save_predictions


__all__ = ['detect_classify']


def detect_classify(detection_model: torch.nn.Module,
                    classification_model: torch.nn.Module,
                    dataset_path: str,
                    detect_transform: callable,
                    classify_transform: callable,
                    drop: List[str],
                    label_map: Dict[str, int],
                    batch_size: int = 64) -> List[Dict[str, Any]]:
    """
    Performs detection and classification on a dataset, saving predictions and returning detailed attack information.

    Parameters:
        detection_model (torch.nn.Module): Model for detection.
        classification_model (torch.nn.Module): Model for classification.
        dataset_path (str): Path to the dataset file.
        detect_transform (callable): Transformations to apply for detection.
        classify_transform (callable): Transformations to apply for classification.
        drop (List[str]): List of columns to drop from the dataset.
        label_map (Dict[str, int]): Mapping from labels to integers.
        batch_size (int, optional): Batch size for dataloaders. Defaults to 64.

    Returns:
        List[Dict[str, Any]]: List of dictionaries with detailed information about detected attacks.
    """

    index_to_label = {index: label for label, index in label_map.items()}

    # Detection model predictions
    data = reload_data(dataset_path, drop, detect_transform, batch_size)
    predictions = get_predictions(detection_model, data.live_dataloader)
    predictions_d = [pred for pred in predictions]
    save_predictions(dataset_path, predictions_d, 'Label')

    # Classification model predictions
    data = reload_data(dataset_path, drop, classify_transform, batch_size)
    predictions = get_predictions(classification_model, data.live_dataloader)
    predictions_c = [index_to_label[pred] for pred in predictions]
    save_predictions(dataset_path, predictions_c, 'Attack')

    # Process predictions to output the desired dictionary
    # data  = _reload_data(dataset_path, drop, batch_size)
    output_dict = []
    for idx, pred in enumerate(predictions_c):
        # Check if the prediction is not benign
        if predictions_d[idx] == 1 and pred.lower() != "Benign".lower():
            sample, _ = data.pure_live_dataset[idx]
            attack_info = {
                "src_address": sample.IPV4_SRC_ADDR,
                "src_port": sample.L4_SRC_PORT,
                "dest_address": sample.IPV4_DST_ADDR,
                "dest_port": sample.L4_DST_PORT,
                "suspected_attack_type": pred
            }
            output_dict.append(attack_info)

    return output_dict
