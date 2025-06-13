"""
InterFormer: Interpretable Transformer-Based Probabilistic Forecasting Model
Based on the paper:
"Interpretable transformer-based model for probabilistic short-term forecasting of residential net load"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax15


# === Local Variable Selection Network (Section 3.1) ===
class LocalVariableSelection(nn.Module):
    def __init__(self, num_vars, d_model, kernel_size):
        super().__init__()
        self.num_vars = num_vars
        self.convs = nn.ModuleList([
            nn.Conv1d(1, d_model, kernel_size=kernel_size, padding=kernel_size-1)
            for _ in range(num_vars)
        ])
        self.linear = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [batch, seq_len, num_vars]
        x = x.permute(0, 2, 1)  # [B, N, T]
        local_states = []
        for i, conv in enumerate(self.convs):
            v = x[:, i:i+1, :]
            out = conv(v)[:, :, :x.size(-1)]
            local_states.append(out.permute(0, 2, 1))  # [B, T, d]
        local_states = torch.stack(local_states, dim=2)  # [B, T, N, d]

        scores = self.linear(local_states).squeeze(-1)  # [B, T, N]
        weights = entmax15(scores, dim=-1)  # [B, T, N]

        weighted = (local_states * weights.unsqueeze(-1)).sum(dim=2)  # [B, T, d]
        return weighted, weights  # return both output and feature importance v_t


# === Sparse Multi-Head Attention with Causal Masking (Section 3.2) ===
class SparseMultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dk = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(self.dk, d_model)

    def forward(self, x):
        B, T, _ = x.size()
        Q = self.W_Q(x).view(B, T, self.num_heads, self.dk).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.num_heads, self.dk).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.num_heads, self.dk).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.dk ** 0.5)

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask[None, None, :, :], float('-inf'))

        weights = entmax15(scores, dim=-1)  # [B, H, T, T]
        weights_avg = weights.mean(dim=1)  # [B, T, T]

        V_combined = V.mean(dim=1)  # [B, T, dk]
        out = torch.matmul(weights_avg, V_combined)  # [B, T, dk]
        return self.W_O(out), weights_avg  # also return attention weights


# === Feedforward Network with Residual and Norm (Eq. 11–12) ===
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
        return self.norm(x + residual)


# === InterFormer Block (Sparse Attention + FFN + Norm) ===
class InterFormerBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads):
        super().__init__()
        self.attn = SparseMultiheadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, attn_weights = self.attn(x)
        x = self.norm(x + attn_out)
        x = self.ffn(x)
        return x, attn_weights


# === InterFormer Model with Logging for Interpretability (Eq. 14-16) ===
class InterFormer(nn.Module):
    def __init__(self, num_vars_cond, num_vars_pred, d_model, kernel_size,
                 num_heads, d_ff, num_layers, horizon, quantiles):
        super().__init__()
        self.quantiles = quantiles
        self.horizon = horizon

        self.selector_cond = LocalVariableSelection(num_vars_cond, d_model, kernel_size)
        self.selector_pred = LocalVariableSelection(num_vars_pred, d_model, kernel_size)

        self.blocks = nn.ModuleList([
            InterFormerBlock(d_model, d_ff, num_heads)
            for _ in range(num_layers)
        ])

        self.projection = nn.ModuleList([
            nn.Linear(d_model, horizon) for _ in quantiles
        ])

    def forward(self, x_cond, x_pred):
        x_cond, v_cond = self.selector_cond(x_cond)
        x_pred, v_pred = self.selector_pred(x_pred)
        x = torch.cat([x_cond, x_pred], dim=1)

        attention_logs = []
        for block in self.blocks:
            x, attn_weights = block(x)
            attention_logs.append(attn_weights)  # List of [B, T, T]

        last = x[:, -1, :]  # final time step
        out = torch.stack([proj(last) for proj in self.projection], dim=-1)  # [B, H, Q]
        return out.transpose(1, 2), v_cond, v_pred, attention_logs  # return all logs


# === Pinball Loss for Quantile Regression (Eq. 18) ===
def pinball_loss(y_true, y_pred, quantiles):
    losses = []
    for i, q in enumerate(quantiles):
        err = y_true - y_pred[:, i, :]
        loss = torch.max((q - 1) * err, q * err)
        losses.append(loss)
    return torch.mean(torch.stack(losses, dim=1))
