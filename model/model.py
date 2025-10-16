import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import math

class PositionalEncoding(nn.Module):
    """Adds positional information to the input sequence."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerFeatureExtractor(nn.Module):
    """Uses a Transformer Encoder to extract temporal features from stock data."""
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.model_dim = model_dim
        self.encoder = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

    def forward(self, src):
        # src shape: [batch_size, seq_len, input_dim]
        src = self.encoder(src) * math.sqrt(self.model_dim)
        # Note: PositionalEncoding expects [seq_len, batch_size, dim], so we permute
        src = src.permute(1, 0, 2)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)
        
        output = self.transformer_encoder(src)
        # We only need the features from the last time step for graph input
        return output[:, -1, :] # -> [batch_size, model_dim]

class StockPredictor(nn.Module):
    """Combines Transformer for temporal features and GAT for relational learning."""
    def __init__(self, input_dim, model_dim, num_heads_transformer, num_layers_transformer,
                 num_heads_gat, num_classes=3, dropout=0.1):
        super().__init__()
        self.transformer = TransformerFeatureExtractor(
            input_dim=input_dim,
            model_dim=model_dim,
            num_heads=num_heads_transformer,
            num_layers=num_layers_transformer,
            dropout=dropout
        )

        self.gat1 = GATConv(model_dim, 8, heads=num_heads_gat, dropout=dropout)
        self.gat2 = GATConv(8 * num_heads_gat, num_classes, heads=1, concat=False, dropout=dropout)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x_ts, edge_index):
        # x_ts is the time series data for all stocks in the batch
        # Shape: [num_stocks, seq_len, num_features]
        
        # 1. Get temporal features from Transformer
        node_features = self.transformer(x_ts) # -> [num_stocks, model_dim]

        # 2. Pass features through GAT for relational learning
        x = self.dropout(node_features)
        x = self.gat1(x, edge_index)
        x = self.elu(x)
        x = self.dropout(x)
        x = self.gat2(x, edge_index) # Output logits for classification

        return x
