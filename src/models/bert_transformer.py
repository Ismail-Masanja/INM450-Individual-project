import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig


__all__ = ['BertTransformer']


class BertTransformer(nn.Module):
    """
    A transformer-based model for classification tasks using a pre-trained DistilBERT model.

    Args:
        input_dim (int): Dimensionality of the input features.
        num_classes (int): Number of output classes for the classification task.
    """

    def __init__(self, input_dim: int, num_classes: int):
        super(BertTransformer, self).__init__()
        # Ensure output dimension matches DistilBERT hidden size === 768
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=768),
            nn.PReLU(num_parameters=768),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=768, out_features=768)
        )

        # Configure Distilled BERT model
        config = DistilBertConfig()
        self.transformer = DistilBertModel(config)
        # Classifier matches the number of classes
        self.classifier = nn.Linear(config.dim, num_classes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            X (torch.Tensor): Input tensor of shape [batch_size, input_dim].

        Returns:
            torch.Tensor: Logits tensor of shape [batch_size, num_classes].
        """
        # Project input features to match DistilBERT's embedding dimension
        X = self.feature_extractor(X)

        # Ensure input shape [batch_size, seq_len, embedding_dim]
        X = X.unsqueeze(1)  # Add dummy seq_len dimensions

        # Transformer expects inputs_embeds of shape [batch_size, seq_len, embedding_dim]
        transformer_output = self.transformer(inputs_embeds=X)

        # Use first token's output for classification
        output = self.classifier(transformer_output.last_hidden_state[:, 0])

        return output
