import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier


__all__ = ['RandomForest']


class RandomForest(torch.nn.Module):
    """
    A PyTorch wrapper for the scikit-learn RandomForestClassifier, allowing it to be used
    within a PyTorch model pipeline. This class initializes a RandomForestClassifier and
    provides methods to fit the model to the data and make predictions in a PyTorch-friendly
    format.

    Note: This class is primarily designed for integrating traditional machine learning models
    from scikit-learn into PyTorch workflows and does not directly support GPU-accelerated training.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = None) -> None:
        """
        Initializes the TorchRandomForest model with the specified number of estimators and maximum depth.

        Args:
            n_estimators (int): The number of trees in the forest.
            max_depth (int, optional): The maximum depth of the trees. If None, then nodes are expanded until
                                       all leaves are pure or until all leaves contain less than min_samples_split samples.
        """
        super(RandomForest, self).__init__()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the RandomForest model on the provided data.

        Args:
            X (np.ndarray): The input features, a NumPy array of shape (n_samples, n_features).
            y (np.ndarray): The target values, a NumPy array of shape (n_samples,).
        """
        self.model.fit(X, y)
        
        print(f'{self.__class__.__name__} fitting complete!.')
        self.is_fitted = True
        

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Performs inference using the fitted RandomForest model on the provided input.

        Args:
            X (torch.Tensor): The input features, a PyTorch tensor of shape (n_samples, n_features).

        Returns:
            torch.Tensor: The predicted probabilities, a PyTorch tensor of shape (n_samples, n_classes).
        """
        if not self.is_fitted:
            raise Exception("Model must be fitted before calling forward.")

        # Ensure input on CPU and convert to NumPy for prediction with sklearn
        X_np = X.cpu().numpy()
        probs = self.model.predict_proba(X_np)

        # Convert predictions back to PyTorch tensor and move to the same device as input
        return torch.tensor(probs, dtype=torch.float32).to(X.device)
