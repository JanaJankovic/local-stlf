"""
InterFormer: Interpretable Transformer-Based Probabilistic Forecasting Model
Based on the paper:
"Interpretable transformer-based model for probabilistic short-term forecasting of residential net load"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax15  # For alpha-entmax (Eq. 5)


# === Local Variable Selection Network (Section 3.1) ===
class LocalVariableSelection(nn.Module):
    def __init__(self, num_vars, d_model, kernel_size):
        super().__init__()
        self.num_vars = num_vars
        self.convs = nn.ModuleList([
            nn.Conv1d(1, d_model, kernel_size=kernel_size, padding=kernel_size-1) for _ in range(num_vars)
        ])
        self.linear = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [batch_size, seq_len, num_vars]
        x = x.permute(0, 2, 1)  # -> [batch, num_vars, seq_len]
        local_states = []
        for i, conv in enumerate(self.convs):
            v = x[:, i:i+1, :]
            out = conv(v)[:, :, :x.size(-1)]  # Eq. 2 (Causal 1D conv)
            local_states.append(out.permute(0, 2, 1))
        local_states = torch.stack(local_states, dim=2)  # [batch, seq_len, num_vars, d_model]

        scores = self.linear(local_states).squeeze(-1)  # Eq. 3
        weights = entmax15(scores, dim=-1)  # Eq. 5: alpha-entmax

        weighted = (local_states * weights.unsqueeze(-1)).sum(dim=2)  # Eq. 6
        return weighted  # [batch, seq_len, d_model]


# === Sparse Self-Attention Mechanism (Section 3.2) ===
class SparseMultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dk = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, _ = x.size()
        Q = self.W_Q(x).view(B, T, self.num_heads, self.dk).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.num_heads, self.dk).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.num_heads, self.dk).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.dk ** 0.5)  # Eq. 8
        weights = entmax15(scores, dim=-1)  # Sparse attention (Eq. 8)
        output = torch.matmul(weights, V)
        output = output.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_O(output)  # Eq. 10


# === Feedforward Network with Residuals (Eq. 11-12) ===
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return self.norm(x + residual)  # Eq. 12


# === InterFormer Block ===
class InterFormerBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads):
        super().__init__()
        self.attn = SparseMultiheadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm(x + self.attn(x))
        x = self.ffn(x)
        return x


# === InterFormer Model (Section 3.3 and 3.4) ===
class InterFormer(nn.Module):
    def __init__(self, num_vars, d_model, kernel_size, num_heads, d_ff, num_layers, horizon, quantiles):
        super().__init__()
        self.quantiles = quantiles
        self.horizon = horizon
        self.selector = LocalVariableSelection(num_vars, d_model, kernel_size)
        self.blocks = nn.ModuleList([InterFormerBlock(d_model, d_ff, num_heads) for _ in range(num_layers)])
        self.projection = nn.ModuleList([nn.Linear(d_model, horizon) for _ in quantiles])  # Eq. 13

    def forward(self, x):
        # x: [batch, t0, num_vars] -> inputs from condition window
        x = self.selector(x)  # [batch, t0, d_model]
        for block in self.blocks:
            x = block(x)  # [batch, t0, d_model]
        last = x[:, -1, :]  # use final state at t0
        out = torch.stack([proj(last) for proj in self.projection], dim=-1)  # [batch, horizon, quantiles]
        return out.transpose(1, 2)  # [batch, quantiles, horizon]


# === Pinball Loss (Eq. 18) ===
def pinball_loss(y_true, y_pred, quantiles):
    # y_true: [batch, horizon]
    # y_pred: [batch, quantiles, horizon]
    losses = []
    for i, q in enumerate(quantiles):
        err = y_true - y_pred[:, i, :]
        loss = torch.max((q - 1) * err, q * err)
        losses.append(loss)
    return torch.mean(torch.stack(losses, dim=1))