import torch
import torch.nn as nn
import torch.nn.functional as F

# === Fully Connected RNN (FCRNN) ===
class FCRNN(nn.Module):
    def __init__(self, input_size, hidden_size=12, output_size=1, dropout=0.2):
        super().__init__()
        self.rnn1 = nn.RNN(input_size, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.rnn2 = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)  # output_size = horizon

    def forward(self, x):
        out, _ = self.rnn1(x)
        out = self.dropout1(out)
        out, _ = self.rnn2(out)
        out = self.dropout2(out)
        out = F.relu(self.fc1(out[:, -1, :]))  # use last time step
        out = self.fc2(out)  # [batch, horizon]
        return out

# === BPNN for combining outputs of S-RNN and P-RNN ===
class BPNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=5, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)  # output_size = 1 (scalar for each step)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))  # Final ReLU!

# === Dual-Input RNN (DIRNN) ===
class DIRNN(nn.Module):
    def __init__(self, seq_input_size, per_input_size, hidden_size=12, bp_hidden_size=5, dropout=0.2, horizon=1):
        super().__init__()
        self.horizon = horizon
        self.s_rnn = FCRNN(input_size=seq_input_size, hidden_size=hidden_size, output_size=horizon, dropout=dropout)
        self.p_rnn = FCRNN(input_size=per_input_size, hidden_size=hidden_size, output_size=horizon, dropout=dropout)
        self.bpnn = BPNN(input_size=2, hidden_size=bp_hidden_size, output_size=1)  # output is scalar per step

    def forward(self, x_seq, x_per):
        # x_seq, x_per: [batch, time, feature]
        s_out = self.s_rnn(x_seq)  # [batch, horizon]
        p_out = self.p_rnn(x_per)  # [batch, horizon]

        # Combine each step of horizon with [s_out, p_out] features
        combined = torch.stack([s_out, p_out], dim=2)  # [batch, horizon, 2]
        combined_flat = combined.reshape(-1, 2)        # [batch * horizon, 2]

        bp_out = self.bpnn(combined_flat).squeeze(-1)  # [batch * horizon]
        return bp_out.view(-1, self.horizon)           # [batch, horizon]