import xgboost as xgb
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Any


__all__ = ['XGBoost']


class XGBoost(torch.nn.Module):
    """
    A PyTorch-compatible for XGBoost classifier, enabling its integration
    into PyTorch workflows. This class provides a way to fit an XGBoost model on data
    and perform predictions in a manner that's compatible with PyTorch tensors.

    Attributes:
        model (xgb.XGBClassifier): The underlying XGBoost model.
        is_fitted (bool): Indicates whether the model has been fitted.
        label_encoder (LabelEncoder): Encodes labels into a range between 0 and n_classes-1.
    """

    def __init__(self, **xgb_params: Any) -> None:
        """
        Initializes the XGBoost class with the given parameters for the XGBClassifier.

        Args:
            **xgb_params (Any): Parameters to initialize the XGBClassifier model.
        """
        super(XGBoost, self).__init__()
        self.model = xgb.XGBClassifier(**xgb_params)
        self.is_fitted = False
        self.label_encoder = LabelEncoder()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the XGBoost model on the provided dataset.

        Args:
            X (np.ndarray): Input features; can be a NumPy array or a PyTorch tensor.
            y (np.ndarray): Target labels; can be a NumPy array or a PyTorch tensor.
        """
        # Convert PyTorch tensors to NumPy arrays if necessary
        if isinstance(X, torch.Tensor):
            X = X.cpu().detach().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().detach().numpy()

        # Use LabelEncoder to transform labels to range 0 to n_classes-1
        y_encoded = self.label_encoder.fit_transform(y)

        self.model.fit(X, y_encoded)
        
        print(f'{self.__class__.__name__} fitting complete!.')
        self.is_fitted = True
        

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Performs prediction on input features using the fitted XGBoost model.

        Args:
            X (torch.Tensor): Input features as a PyTorch tensor.

        Returns:
            torch.Tensor: The predicted probabilities for each class as a PyTorch tensor.
        """
        # Ensure the model is fitted
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling forward.")

        if isinstance(X, torch.Tensor):
            device = X.device
            X = X.cpu().detach().numpy()  # Convert to NumPy for prediction
        else:
            # Default to "cpu" device
            device = torch.device("cpu")

        # Make predictions
        predictions = self.model.predict(X)

        # Convert predictions to one-hot encoded format
        num_classes = len(self.label_encoder.classes_)
        one_hot_predictions = np.zeros(
            (predictions.size, num_classes), dtype=np.float32)
        one_hot_predictions[np.arange(predictions.size), predictions] = 1

        # Convert one-hot encoded predictions back to PyTorch tensor, ensure it's on original device
        return torch.from_numpy(one_hot_predictions).to(device)
