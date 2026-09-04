"""
Transformer Encoder Architecture for Multi-Modal Solar Power Forecasting.
Input: (batch_size, seq_len=96, input_dim=13)
Output: (batch_size, horizon=4) -> 1-hour ahead (15, 30, 45, 60 min)
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding injected additively into the input embeddings.
    Provides the Transformer with temporal ordering information since
    self-attention is permutation-invariant by default.
    """

    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class SolarTransformer(nn.Module):
    """
    Transformer Encoder Network for Solar AC Power Nowcasting.
    Features:
    - Linear input projection to d_model dimensions
    - Sinusoidal positional encoding for temporal ordering
    - Multi-layer Transformer Encoder with multi-head self-attention
    - Multi-layer perceptron projection head from final time step
    """

    def __init__(
        self,
        input_dim: int = 13,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        output_dim: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim

        # Project raw features into d_model space
        self.input_projection = nn.Linear(input_dim, d_model)

        # Sinusoidal positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=200, dropout=dropout)

        # Transformer Encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # MLP projection head (from final time step representation)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len=96, input_dim=13)

        # 1. Project to d_model dimensions
        x = self.input_projection(x)  # (batch, seq_len, d_model)

        # 2. Add positional encoding
        x = self.pos_encoder(x)  # (batch, seq_len, d_model)

        # 3. Transformer Encoder (self-attention over full sequence)
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)

        # 4. Extract final time step representation h_T
        last_step = x[:, -1, :]  # (batch, d_model)

        # 5. Predict multi-step horizon
        out = self.head(last_step)  # (batch, output_dim=4)
        return out


if __name__ == "__main__":
    dummy_x = torch.randn(8, 96, 13)
    model = SolarTransformer(
        input_dim=13, d_model=64, nhead=4, num_layers=2,
        dim_feedforward=128, output_dim=4,
    )
    out = model(dummy_x)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Transformer Model instantiated successfully. Output shape: {out.shape}")
    print(f"Total trainable parameters: {total_params:,}")
