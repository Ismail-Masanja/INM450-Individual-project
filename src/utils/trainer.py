from typing import Callable, Optional
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module


__all__ = ['Trainer']


class Trainer:
    """
    A class for training different pytorch models inheriting from torch.nn.modules.

    Attributes:
        training_data (DataLoader): DataLoader for the training dataset.
        testing_data (DataLoader): DataLoader for the testing dataset.
        optimizer (Optimizer): The optimization algorithm.
        costfx (Callable): The loss function.
        device (torch.device): Device on which to train ('cuda' or 'cpu').
        epochs (int): Number of training epochs.
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Learning rate scheduler.

    Methods:
        train(model: Module): Trains the given model using the initialized parameters.
    """

    def __init__(self,
                 training_data: DataLoader,
                 testing_data: DataLoader,
                 optimizer: Optimizer,
                 costfx: Callable,
                 device: torch.device,
                 epochs: int = 10,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> None:
        """
        Initializes the trainer with training data, optimization strategy, and training parameters.

        Args:
            training_data (DataLoader): DataLoader for the training dataset.
            testing_data (DataLoader): DataLoader for the testing dataset.
            optimizer (Optimizer): The optimization algorithm.
            costfx (Callable): The loss function.
            device (torch.device): Device on which to train ('cuda' or 'cpu').
            epochs (int): Number of training epochs.
            scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Learning rate scheduler.
        """
        self.trnd = training_data
        self.tstd = testing_data
        self.epochs = epochs
        self.optim = optimizer
        self.costfx = costfx
        self.device = device
        self.scheduler = scheduler

    def train(self, model: Module) -> None:
        """
        Trains the model using the training data and updates the model's weights.

        Args:
            model (Module): The neural network model to be trained.
        """
        model.to(self.device)
        training_losses = []
        testing_losses = []

        for epoch in tqdm(range(self.epochs), desc='Training Epochs'):
            model.train()
            running_loss = 0.0
            for inputs, targets in self.trnd:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optim.zero_grad()
                outputs = model(inputs)
                loss = self.costfx(outputs, targets)
                loss.backward()
                self.optim.step()

                running_loss += loss.item()

            if self.scheduler is not None:
                self.scheduler.step()

            avg_train_loss = running_loss / len(self.trnd)
            training_losses.append(avg_train_loss)

            model.eval()
            running_loss = 0.0
            with torch.no_grad():
                for inputs, targets in self.tstd:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    loss = self.costfx(outputs, targets)
                    running_loss += loss.item()

            avg_test_loss = running_loss / len(self.tstd)
            testing_losses.append(avg_test_loss)

            print(
                f"Epoch {epoch+1}/{self.epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_test_loss:.4f}")

        self._plot_losses(training_losses, testing_losses, model)

    def _plot_losses(self, training_losses: list, testing_losses: list, model: Module) -> None:
        """
        Plots the training and testing losses over epochs.

        Args:
            training_losses (list): List of training losses per epoch.
            testing_losses (list): List of testing/validation losses per epoch.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, self.epochs+1), training_losses, label='Training Loss')
        plt.plot(range(1, self.epochs+1), testing_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.xticks(np.arange(1, self.epochs + 1, 1.0))
        plt.title(f'{model.__class__.__name__} Training and Validation Losses')
        plt.legend()
        plt.show()
