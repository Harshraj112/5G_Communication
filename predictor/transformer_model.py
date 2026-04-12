"""
transformer_model.py — Encoder-only Transformer for 5G traffic demand forecasting.

Input  : (batch, T, 3)   — last T timesteps of [eMBB, URLLC, mMTC] demand
Output : (batch, H, 3)   — next H timestep forecasts
"""

import math
import torch
import torch.nn as nn


# ─────────────────────────────────────────────
# Positional Encoding
# ─────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as in "Attention Is All You Need".
    Adds position information to token embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)                       # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float() # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                                       # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ─────────────────────────────────────────────
# Transformer Traffic Predictor
# ─────────────────────────────────────────────

class TrafficTransformer(nn.Module):
    """
    Encoder-only Transformer that maps a window of T past observations
    to H future demand forecasts.

    Architecture:
        Linear projection → Positional Encoding → N×TransformerEncoderLayer
        → Linear projection → Output reshape

    Args:
        n_features : Number of traffic slices (default 3)
        d_model    : Internal embedding dimension
        n_heads    : Number of attention heads
        n_layers   : Number of encoder layers
        dropout    : Dropout probability
        window_t   : Input sequence length T
        horizon_h  : Output forecast length H
    """

    def __init__(
        self,
        n_features: int = 3,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        window_t: int = 20,
        horizon_h: int = 10,
    ):
        super().__init__()
        self.window_t  = window_t
        self.horizon_h = horizon_h
        self.n_features = n_features
        self.d_model = d_model

        # Input projection: 3 → d_model
        self.input_proj = nn.Linear(n_features, d_model)

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_len=window_t + 10, dropout=dropout)

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,   # (batch, seq, d_model)
            norm_first=False,   # Post-LN (avoids PyTorch nested-tensor warning)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output head: mean-pooled encoder output → (H * n_features)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, horizon_h * n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, T, n_features) — normalised input window
        Returns:
            out: (batch, H, n_features) — forecast
        """
        batch = x.size(0)

        # Project input to d_model
        x = self.input_proj(x)              # (batch, T, d_model)
        x = self.pos_enc(x)                 # (batch, T, d_model)

        # Encode
        x = self.encoder(x)                 # (batch, T, d_model)

        # Mean pool and project to output
        x = x.mean(dim=1)                   # (batch, d_model)
        x = self.output_head(x)             # (batch, H*n_features)
        x = x.view(batch, self.horizon_h, self.n_features)  # (batch, H, 3)

        return x


# ─────────────────────────────────────────────
# Convenience loader
# ─────────────────────────────────────────────

def load_model(path: str, device: torch.device | str = "cpu", **kwargs) -> TrafficTransformer:
    """Load a saved TrafficTransformer from `path`."""
    model = TrafficTransformer(**kwargs)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model
