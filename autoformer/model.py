import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        trend = self.avg_pool(x)
        if trend.shape[-1] != x.shape[-1]:
            trend = F.pad(trend, (0, x.shape[-1] - trend.shape[-1]))
        seasonal = x - trend
        return seasonal, trend


class AutoCorrelation(nn.Module):
    def __init__(self, top_k=3):
        super().__init__()
        self.top_k = top_k

    def forward(self, x):
        B, C, T = x.size()
        fft_x = torch.fft.rfft(x, dim=-1)
        auto_corr = torch.fft.irfft(fft_x * torch.conj(fft_x), n=T, dim=-1)
        lags = torch.topk(auto_corr, self.top_k, dim=-1).indices

        agg = torch.zeros_like(x)
        for k in range(self.top_k):
            lag = lags[..., k]
            for b in range(B):
                for c in range(C):
                    shift = lag[b, c].item()
                    if shift == 0:
                        agg[b, c] += x[b, c]
                    else:
                        agg[b, c, shift:] += x[b, c, :-shift]
        agg /= self.top_k
        return agg


class AutoformerBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=25, top_k=3):
        super().__init__()
        self.decomp = SeriesDecomposition(kernel_size)
        self.auto_corr = AutoCorrelation(top_k)
        self.proj = nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        seasonal, trend = self.decomp(x)
        seasonal = self.auto_corr(seasonal)
        return self.proj(seasonal + trend), trend, seasonal


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, d_model, forecast_horizon):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(d_model, in_channels * forecast_horizon, kernel_size=1)
        )

    def forward(self, x, forecast_horizon):
        B, C, T = x.shape
        out = self.layer(x)  # [B, C*H, T]
        out = out.view(B, C, forecast_horizon, T)[:, :, :, -1]  # [B, C, H]
        return out


class MultiFactorAttention(nn.Module):
    def __init__(self, d_model, num_factors):
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(num_factors, d_model)
        self.value_proj = nn.Linear(num_factors, d_model)
        self.score_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** 0.5

    def forward(self, y_hat, weather_hist, weather_fore):
        # Concatenate weather inputs: [B, H, D]
        weather = torch.cat([weather_hist, weather_fore], dim=1)  # [B, H_total, F]
        Q = self.query_proj(y_hat)                      # [B, H, D]
        K = self.key_proj(weather)                      # [B, H_total, D]
        V = self.value_proj(weather)                    # [B, H_total, D]

        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # [B, H, H_total]
        attn_weights = F.softmax(attn_scores, dim=-1)               # [B, H, H_total]
        correction = torch.bmm(attn_weights, V)                     # [B, H, D]

        combined = self.score_proj(correction)                     # [B, H, D]
        P = F.softmax(torch.sum(combined * y_hat, dim=-1, keepdim=True) / self.scale, dim=1)  # [B, H, 1]
        final = torch.sum(P * y_hat, dim=1, keepdim=True)                                    # [B, 1, D]
        return y_hat + final.expand_as(y_hat)                      # [B, H, D]


class AutoformerForecast(nn.Module):
    def __init__(self, in_channels=1, d_model=64, num_factors=5, top_k=3, kernel_size=25, forecast_horizon=24):
        super().__init__()
        self.forecast_horizon = forecast_horizon

        # Encoder: stack 2 Autoformer blocks
        self.encoder = nn.ModuleList([
            AutoformerBlock(in_channels, kernel_size, top_k),
            AutoformerBlock(in_channels, kernel_size, top_k)
        ])

        # Decoder blocks for both trend and seasonal components
        self.trend_decoder = DecoderBlock(in_channels, d_model, forecast_horizon)
        self.seasonal_decoder = DecoderBlock(in_channels, d_model, forecast_horizon)

        # Multi-Factor Attention
        self.mfa = MultiFactorAttention(d_model, num_factors)

    def forward(self, x, weather_hist, weather_fore):
        B, C, T = x.shape

        seasonal, trend, _ = self.encoder[0](x)
        seasonal, trend, seasonal2 = self.encoder[1](seasonal + trend)

        trend_out = self.trend_decoder(trend, self.forecast_horizon)        # [B, C, H]
        seasonal_out = self.seasonal_decoder(seasonal2, self.forecast_horizon)  # [B, C, H]

        decoded = trend_out + seasonal_out         # [B, C, H]
        decoded = decoded.permute(0, 2, 1)         # [B, H, C]

        corrected = self.mfa(decoded, weather_hist, weather_fore)  # [B, H, C]
        return corrected.permute(0, 2, 1)          # [B, C, H]
