
import torch
import torch.nn as nn
from typing import Tuple
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────────────────────────────────── Mixer building block ────────────────
class MLPOnlyLOB(nn.Module):
    """
    One token-mixing + one channel-mixing MLP block with residual connections.
    Input / output shape: (B, T, C)  —  T = seq_len, C = feature_dim
    """
    def __init__(
        self,
        seq_len: int,
        feature_dim: int,
        channel_hidden: int = 512,
        token_hidden: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        # LayerNorm is applied over last dim by default
        self.norm_chan = nn.LayerNorm(feature_dim)
        self.mlp_chan = nn.Sequential(
            nn.Linear(feature_dim, channel_hidden, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(channel_hidden, feature_dim, bias=False),
            nn.Dropout(dropout),
        )

        self.norm_tok = nn.LayerNorm(seq_len)
        self.mlp_tok = nn.Sequential(
            nn.Linear(seq_len, token_hidden, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(token_hidden, seq_len, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── reshape to (B, T, C) ─────────────────────────
        if x.ndim == 4:                       # (B, 1, T, C)
            if x.size(1) != 1:
                raise ValueError(f"Channel dim ≠ 1: got {x.shape}")
            x = x.squeeze(1)                  # → (B, T, C)
        elif x.ndim != 3:
            raise ValueError(f"Expected 3-D or 4-D input, got {x.shape}")

        # ----- Channel-mix (per token) -------------------
        y = self.norm_chan(x)                 # works: last dim = C
        y = self.mlp_chan(y)
        x = x + y

        # ----- Token-mix (per channel) -------------------
        y = self.norm_tok(x.transpose(1, 2))  # now last dim = T = 100
        y = self.mlp_tok(y)
        x = x + y.transpose(1, 2)
        return x
