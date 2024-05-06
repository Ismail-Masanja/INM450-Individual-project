import torch
import torch.nn as nn
import torch.nn.functional as F
import math


__all__ = ['TransformerEncoderClassifier']


class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to the input tensor to incorporate information about the position of the tokens.

    Args:
        d_model (int): Dimensionality of the input embedding.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        max_len (int, optional): Maximum length of the input sequences. Defaults to 5000.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Change shape to [max_len, 1, d_model] for broadcasting
        pe = pe.unsqueeze(0).transpose(0, 1)
        # Register positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for adding positional encoding.

        Args:
            X (torch.Tensor): Input tensor with shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor with positional encoding added.
        """
        # X [batch_size, seq_len, d_model]
        X = X + self.pe[:X.size(1), :]  # broadcast pe to match X
        return self.dropout(X)


class TransformerEncoderClassifier(nn.Module):
    """
    A Transformer Encoder based classifier.

    Args:
        input_dim (int): Dimensionality of the input feature vector.
        d_model (int): Dimensionality of the model's input embedding.
        nhead (int): Number of heads in the multiheadattention models.
        num_encoder_layers (int): Number of sub-encoder-layers in the encoder.
        num_classes (int): Number of output classes.
        dim_feedforward (int, optional): Dimension of the feedforward network model. Defaults to 2048.
        dropout (float, optional): Dropout value. Defaults to 0.1.
    """
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_encoder_layers: int, num_classes: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super(TransformerEncoderClassifier, self).__init__()
        
        # Linear layer to project input sequence to expected dimension
        # Replacement for nn.Embedding.
        self.input_projection = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=d_model),
            nn.LayerNorm(d_model),
            nn.PReLU(num_parameters=d_model),
            nn.Dropout(p=0.1),  
            nn.Linear(in_features=d_model, out_features=d_model)
        )
        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        self.d_model = d_model
        
        # self.classifier = nn.Linear(d_model, num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_model),
            nn.LayerNorm(d_model),
            nn.PReLU(num_parameters=d_model),
            nn.Dropout(p=0.1),  
            nn.Linear(in_features=d_model, out_features=num_classes)
        )

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the Transformer Encoder Classifier.

        Args:
            src (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
            src_mask (torch.Tensor, optional): Source mask tensor. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        src = self.input_projection(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        output = self.transformer_encoder(src, src_mask)

        output = output.permute(1, 0, 2)
        output = self.classifier(output[:, -1])

        return output