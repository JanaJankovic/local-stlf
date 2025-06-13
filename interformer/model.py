"""
InterFormer: Interpretable Transformer-Based Probabilistic Forecasting Model
Based on the paper:
"Interpretable transformer-based model for probabilistic short-term forecasting of residential net load"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax15


# === Section 3.1: Local Variable Selection Network ===
class LocalVariableSelection(nn.Module):
    def __init__(self, num_vars, d_model, kernel_size):
        super().__init__()
        self.num_vars = num_vars
        self.convs = nn.ModuleList([
            nn.Conv1d(1, d_model, kernel_size=kernel_size, padding=kernel_size - 1)
            for _ in range(num_vars)
        ])
        self.linear = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [B, T, N] → conv needs [B, 1, T] per variable
        x = x.permute(0, 2, 1)  # [B, N, T]
        local_states = []

        for i, conv in enumerate(self.convs):
            v = x[:, i:i + 1, :]  # [B, 1, T]
            out = conv(v)[:, :, :x.size(-1)]  # crop extra padding
            local_states.append(out.permute(0, 2, 1))  # [B, T, d]

        local_states = torch.stack(local_states, dim=2)  # [B, T, N, d]
        scores = self.linear(local_states).squeeze(-1)  # [B, T, N]
        min_score = scores.min().item()
        max_score = scores.max().item()
        mean_score = scores.mean().item()
        std_score = scores.std().item()

        if torch.isnan(scores).any() or std_score < 1e-6:
            print("⚠️ Unstable attention scores detected:")
            print(f"  → min: {min_score:.6f}, max: {max_score:.6f}, mean: {mean_score:.6f}, std: {std_score:.6f}")
            print(f"  → Raw score slice (scores[0, 0, :]): {scores[0, 0, :].detach().cpu().numpy()}")

        # === Numerical safety for entmax15 ===
        # Clamp extreme values
        scores = scores.clamp(min=-10, max=10)

        # Add small noise to avoid ties / zero-variance
        eps = 1e-4
        noise = eps * torch.randn_like(scores)
        scores = scores + noise

        # Sparse attention
        weights = entmax15(scores, dim=-1)  # [B, T, N]
        weighted = (local_states * weights.unsqueeze(-1)).sum(dim=2)  # [B, T, d]
        return weighted, weights



# === Section 3.2: Sparse Multi-Head Attention with Causal Masking ===
class SparseMultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dk = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(self.dk, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.size()
        Q = self.W_Q(x).view(B, T, self.num_heads, self.dk).transpose(1, 2)  # [B, H, T, d_k]
        K = self.W_K(x).view(B, T, self.num_heads, self.dk).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.num_heads, self.dk).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.dk ** 0.5)  # [B, H, T, T]
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask[None, None, :, :], float('-inf'))

        weights = entmax15(scores, dim=-1)  # [B, H, T, T]
        weights = self.dropout(weights)  # Dropout on attention weights (optional)
        weights_avg = weights.mean(dim=1)  # [B, T, T]

        V_combined = V.mean(dim=1)  # mean over heads → [B, T, d_k]
        out = torch.matmul(weights_avg, V_combined)  # [B, T, d_k]
        return self.W_O(out), weights_avg  # [B, T, d_model], attention


# === Eq. 11–12: Feedforward Network with Residual & Normalization ===
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return self.norm(x + residual)


# === InterFormer Block: Sparse Attention + FFN + Norm ===
class InterFormerBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout=0.0):
        super().__init__()
        self.attn = SparseMultiheadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, attn_weights = self.attn(x)
        x = self.norm(x + attn_out)
        x = self.ffn(x)
        return x, attn_weights


# === Eq. 14–16: Final InterFormer Architecture ===
class InterFormer(nn.Module):
    def __init__(self, num_vars_cond, num_vars_pred, d_model, kernel_size,
                 num_heads, d_ff, num_layers, horizon, quantiles, dropout=0.0):
        super().__init__()
        self.quantiles = quantiles
        self.horizon = horizon

        # Input selection
        self.selector_cond = LocalVariableSelection(num_vars_cond, d_model, kernel_size)
        self.selector_pred = LocalVariableSelection(num_vars_pred, d_model, kernel_size)

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            InterFormerBlock(d_model, d_ff, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Output projections: one per quantile
        self.projection = nn.ModuleList([
            nn.Linear(d_model, horizon) for _ in quantiles
        ])

    def forward(self, x_cond, x_pred):
        # 1. Local feature selection
        x_cond, v_cond = self.selector_cond(x_cond)
        x_pred, v_pred = self.selector_pred(x_pred)

        # 2. Sequence modeling
        x = torch.cat([x_cond, x_pred], dim=1)
        attention_logs = []
        for block in self.blocks:
            x, attn_weights = block(x)
            attention_logs.append(attn_weights)

        # 3. Multi-step quantile prediction
        # Option: Use the last timestep's features for projection
        seq_last = x[:, -1, :]  # [B, d_model]
        out = torch.stack([proj(seq_last) for proj in self.projection], dim=1)  # [B, Q, H]
        
        return out, v_cond, v_pred, attention_logs



# === Eq. 18: Pinball Loss for Quantile Regression ===
def pinball_loss(y_true, y_pred, quantiles):
    losses = []
    for i, q in enumerate(quantiles):
        err = y_true - y_pred[:, i, :]
        loss = torch.max((q - 1) * err, q * err)
        losses.append(loss)
    return torch.mean(torch.stack(losses, dim=1))
