"""
GRU Neural Network Architecture for Multi-Modal Solar Power Forecasting.
Input: (batch_size, seq_len=96, input_dim=13)
Output: (batch_size, horizon=4) -> 1-hour ahead (15, 30, 45, 60 min)
"""

import torch
import torch.nn as nn


class SolarGRU(nn.Module):
    """
    Stacked Gated Recurrent Unit (GRU) Network for Solar AC Power Nowcasting.
    Features:
    - 2-layer stacked GRU cell with reset and update gates
    - Inter-layer dropout for regularization
    - Multi-layer perceptron projection head
    """

    def __init__(
        self,
        input_dim: int = 13,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len=96, input_dim=13)
        gru_out, _ = self.gru(x)
        # Extract representation from final time step h_T
        last_step = gru_out[:, -1, :]
        # Predict multi-step horizon (batch_size, output_dim=4)
        out = self.head(last_step)
        return out


if __name__ == "__main__":
    dummy_x = torch.randn(8, 96, 13)
    model = SolarGRU(input_dim=13, hidden_dim=64, num_layers=2, output_dim=4)
    out = model(dummy_x)
    print("GRU Model instantiated successfully. Output shape:", out.shape)
