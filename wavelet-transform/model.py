import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt


class Time2Vec(nn.Module):
    def __init__(self, input_size, k):
        super(Time2Vec, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.periodic = nn.Linear(input_size, k - 1)

    def forward(self, x):
        linear_term = self.linear(x)
        periodic_term = torch.sin(self.periodic(x))
        return torch.cat([linear_term, periodic_term], dim=-1)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout1(attn_output))
        x = self.norm2(x + self.dropout2(self.ff(x)))
        return x


class SwtTransformerCore(nn.Module):
    def __init__(self, input_size, time2vec_k, d_model, n_heads, d_ff, n_enc_layers, output_size):
        super().__init__()
        self.time2vec = Time2Vec(input_size, time2vec_k)
        self.proj = nn.Linear(time2vec_k, d_model)

        self.encoder_stack = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_enc_layers)
        ])

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, output_size)
        )

    def forward(self, x):
        x = self.time2vec(x)
        x = self.proj(x)

        for enc_layer in self.encoder_stack:
            x = enc_layer(x)

        x_out = self.global_pool(x.transpose(1, 2)).squeeze(-1)
        return self.head(x_out)


class SwtForecastingModel(nn.Module):
    def __init__(self, num_bands, *args, **kwargs):
        super().__init__()
        self.models = nn.ModuleList([
            SwtTransformerCore(*args, **kwargs) for _ in range(num_bands)
        ])

    def forward(self, swt_inputs):
        # swt_inputs: list of [B, T, 1] tensors for each sub-band (A3, D1, D2, D3)
        outputs = []
        for model, x in zip(self.models, swt_inputs):
            outputs.append(model(x))  # [B, 1] or [B, output_size]
        return torch.stack(outputs, dim=1)  # [B, num_bands, output_size]


# === SWT UTILS ===
def swt_decompose(signal_np, wavelet='db2', level=3):
    coeffs = pywt.swt(signal_np, wavelet, level=level)
    return list(reversed(coeffs))  # [(A3, D3), (A2, D2), (A1, D1)]


def swt_reconstruct(coeffs, wavelet='db2'):
    coeffs = list(reversed(coeffs))
    return pywt.iswt(coeffs, wavelet)
