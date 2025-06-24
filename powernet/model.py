import torch
import torch.nn as nn
import torch.nn.functional as F

class PowerNet(nn.Module):
    def __init__(self, input_size_meta, horizon, input_size_lstm=1, lstm_hidden=128, lstm_layers=2,
                 mlp_hidden1=64, mlp_hidden2=32, final_hidden=64, dropout=0.2):
        super(PowerNet, self).__init__()

        self.horizon = horizon  # save for shape consistency

        # LSTM branch for sequential consumption data
        self.lstm = nn.LSTM(
            input_size=input_size_lstm,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        # MLP branch for weather + calendar features
        self.mlp = nn.Sequential(
            nn.Linear(input_size_meta, mlp_hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden1, mlp_hidden2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Final fusion and prediction layer
        self.predictor = nn.Sequential(
            nn.Linear(lstm_hidden + mlp_hidden2, final_hidden),
            nn.ReLU(),
            nn.Linear(final_hidden, horizon)  # Predict full horizon
        )

    def forward(self, seq_x, meta_x):
        # LSTM encoding
        lstm_out, _ = self.lstm(seq_x)  # lstm_out: [B, T, H]
        seq_encoded = lstm_out[:, -1, :]  # Take last time step [B, H]

        # MLP encoding
        meta_encoded = self.mlp(meta_x)  # [B, H]

        # Fusion and prediction
        combined = torch.cat((seq_encoded, meta_encoded), dim=1)  # [B, H + H]
        out = self.predictor(combined)  # [B, horizon]

        return out  # [B, horizon]