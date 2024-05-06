import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


__all__ = ['ModelEvaluator']


class ModelEvaluator:
    """
    A class for evaluating a PyTorch model on a given dataset.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset to be evaluated.
        labels_map (dict): A mapping from numeric labels to their corresponding string labels.
        device (str, optional): The device to perform evaluation on. Defaults to 'cpu'.
    """

    def __init__(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, labels_map: dict, device: str = 'cpu'):
        self.model = model
        self.dataloader = dataloader
        # Map numeric labels back to string labels
        self.labels_map = {v: k for k, v in labels_map.items()}
        self.device = device

    def evaluate(self) -> None:
        """
        Evaluates the model on the dataset and prints performance metrics and a confusion matrix.
        """
        y_true = []
        y_pred = []

        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)

                y_true.extend(labels.detach().cpu().numpy())
                y_pred.extend(predicted.detach().cpu().numpy())

        # Calculate, print Metrics Summary
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        print(f'{self.model.__class__.__name__} Metrics Summary')
        print(
            f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

        # Generate, print classification report
        class_report = classification_report(
            y_true, y_pred, target_names=list(self.labels_map.values()), zero_division=0)
        print(f'\n{self.model.__class__.__name__} Classification Report:\n', class_report)

        # Plot, show confusion matrix
        self._plot_confusion_matrix(y_true, y_pred)

    def _plot_confusion_matrix(self, y_true: list, y_pred: list) -> None:
        """
        Plots the confusion matrix for the given true and predicted labels.

        Args:
            y_true (list): List of true labels.
            y_pred (list): List of predicted labels.
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(
            self.labels_map.values()), yticklabels=list(self.labels_map.values()))
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(f'{self.model.__class__.__name__} Confusion Matrix')
        plt.show()
