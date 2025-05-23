import torch
import torch.nn as nn
import torch.nn.functional as F

# === Fully Connected Recurrent Neural Network (Section 2.2, Eq. 4–6) ===
class FCRNN(nn.Module):
    def __init__(self, input_size, hidden_size=12, output_size=1, dropout=0.2):
        super().__init__()
        self.rnn1 = nn.RNN(input_size, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.rnn2 = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn1(x)
        out = self.dropout1(out)
        out, _ = self.rnn2(out)
        out = self.dropout2(out)
        out = F.relu(self.fc1(out[:, -1, :]))
        out = F.relu(self.fc2(out))
        return out  # [batch, 1]

# === Backpropagation Neural Network (BPNN) for combining S-RNN and P-RNN (Eq. 10) ===
class BPNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))

# === Dual Granularity Inputs Structured RNNs Ensemble (DI-RNN, Section 2.3) ===
class DIRNN(nn.Module):
    def __init__(self, seq_input_size, per_input_size, hidden_size=12, bp_hidden_size=5, dropout=0.2):
        super().__init__()
        self.s_rnn = FCRNN(input_size=seq_input_size, hidden_size=hidden_size, dropout=dropout)
        self.p_rnn = FCRNN(input_size=per_input_size, hidden_size=hidden_size, dropout=dropout)
        self.bpnn = BPNN(input_size=2, hidden_size=bp_hidden_size)

    def forward(self, x_seq, x_per):
        # x_seq: [batch, m, 1], x_per: [batch, n, 1]
        s_out = self.s_rnn(x_seq).squeeze(-1)  # [batch]
        p_out = self.p_rnn(x_per).squeeze(-1)  # [batch]
        combined = torch.stack([s_out, p_out], dim=1)  # [batch, 2]
        return self.bpnn(combined).squeeze(-1)  # [batch]
