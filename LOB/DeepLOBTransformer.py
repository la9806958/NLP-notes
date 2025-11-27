import torch
import torch.nn as nn
import math

class DeepLOBTransformer(nn.Module):
    def __init__(self, y_len, seq_len=100, feature_dim_in=120):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim_in = feature_dim_in
        self.feature_dim = 192  # Dimensionality after projection
        self.hidden_size = 64   # LSTM hidden size

        # Simplified MLP: single projection layer
        self.feature_proj = nn.Linear(feature_dim_in, self.feature_dim)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True
        )

        # Final output layer
        self.fc = nn.Linear(self.hidden_size, y_len)

    def forward(self, x):
        # Input: (batch, 1, 100, 98)
        if x.ndim == 4 and x.size(1) == 1:
            x = x.squeeze(1)  # → (batch, 100, 98)
        else:
            raise ValueError(f"Expected input of shape (B, 1, 100, 98), got {x.shape}")
    
        x = self.feature_proj(x)      # (batch, 100, 192)
        x, _ = self.lstm(x)           # (batch, 100, 64)
        return self.fc(x[:, -1])      # (batch, y_len)

