import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

class Time2Vec(nn.Module):
    def __init__(self, input_dim, k):
        super().__init__()
        self.w0 = nn.Linear(input_dim, 1)
        self.wp = nn.Linear(input_dim, k - 1)

    def forward(self, x):
        v0 = self.w0(x)
        vp = torch.sin(self.wp(x))
        return torch.cat([v0, vp], dim=-1)

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
        self.dropout_attn = nn.Dropout(0.1)
        self.norm_attn = nn.LayerNorm(d_model)
        self.ff1 = FeedForward(d_model, d_ff)
        self.dropout_ff1 = nn.Dropout(0.1)
        self.norm_ff1 = nn.LayerNorm(d_model)
        self.ff2 = FeedForward(d_model, d_ff)
        self.dropout_ff2 = nn.Dropout(0.1)
        self.norm_ff2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm_attn(x + self.dropout_attn(attn_output))
        ff1_output = self.ff1(x)
        x = self.norm_ff1(x + self.dropout_ff1(ff1_output))
        ff2_output = self.ff2(x)
        x = self.norm_ff2(x + self.dropout_ff2(ff2_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True)
        self.dropout_attn = nn.Dropout(0.1)
        self.norm_attn = nn.LayerNorm(d_model)
        self.ff1 = FeedForward(d_model, d_ff)
        self.dropout_ff1 = nn.Dropout(0.1)
        self.norm_ff1 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm_attn(x + self.dropout_attn(attn_output))
        ff1_output = self.ff1(x)
        x = self.norm_ff1(x + self.dropout_ff1(ff1_output))
        return x

class SwtTransformerCore(nn.Module):
    def __init__(self, input_size, time2vec_k, d_model, n_heads, d_ff, n_enc_layers, n_dec_layers, forecast_steps, output_bands):
        super().__init__()
        self.forecast_steps = forecast_steps
        self.output_bands = output_bands

        self.time2vec = Time2Vec(input_size, time2vec_k)
        self.proj = nn.Linear(time2vec_k, d_model)

        self.encoder_stack = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_enc_layers)
        ])
        self.decoder_stack = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_dec_layers)
        ])

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, forecast_steps)
        )

    def forward(self, x):
        B = x.size(0)
        x = self.time2vec(x)
        x = self.proj(x)

        for enc_layer in self.encoder_stack:
            x = enc_layer(x)
        for dec_layer in self.decoder_stack:
            x = dec_layer(x)

        x_out = self.global_pool(x.transpose(1, 2)).squeeze(-1)
        x_out = self.head(x_out)
        return x_out.view(x.size(0), self.forecast_steps, 1)  # ✅ [B*N, s, bands]

class SwtForecastingModel(nn.Module):
    def __init__(self, input_size, time2vec_k, d_model, n_heads, d_ff, n_enc_layers, n_dec_layers, forecast_steps, output_bands):
        super().__init__()
        self.core = SwtTransformerCore(
            input_size, time2vec_k, d_model, n_heads, d_ff,
            n_enc_layers, n_dec_layers, forecast_steps, output_bands
        )

    def forward(self, x):
        # x: [B, bands, W, 1]
        B, bands, W, _ = x.shape
        x = x.view(B * bands, W, 1)  # Flatten bands into batch dimension

        output = self.core(x)  # [B * bands, s, 1]
        return output.view(B, bands, self.core.forecast_steps, 1)


def swt_decompose(signal_np, wavelet='db2', level=3):
    coeffs = pywt.swt(signal_np, wavelet, level=level)
    return list(reversed(coeffs))

def swt_reconstruct(coeffs, wavelet='db2'):
    coeffs = list(reversed(coeffs))
    return pywt.iswt(coeffs, wavelet)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_state_dict = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
