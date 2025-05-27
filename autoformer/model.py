import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
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
            nn.Conv1d(d_model, d_model * forecast_horizon, kernel_size=1)
        )

    def forward(self, x, forecast_horizon):
        B, C, T = x.shape
        out = self.layer(x)
        try:
            out = out.view(B, -1, forecast_horizon, T)
            out = out[:, :, :, -1]
            return out
        except RuntimeError as e:
            print("Shape mismatch:", out.shape, f"B={B}, C={C}, H={forecast_horizon}")
            raise e


class MultiFactorAttention(nn.Module):
    def __init__(self, d_model, num_factors):
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(1, d_model)  # Feature-wise scoring
        self.value_proj = nn.Linear(1, d_model)
        self.score_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** 0.5

    def forward(self, y_hat, weather_hist, weather_fore):
        B, H, D = y_hat.shape
        full_weather = torch.cat([weather_hist, weather_fore], dim=1)  # [B, H_total, F]
        F_dim = full_weather.shape[-1]

        # Aggregate each feature separately
        weather_scores = []
        for i in range(F_dim):
            feature = full_weather[:, :, i:i+1]  # [B, H_total, 1]
            K = self.key_proj(feature)
            V = self.value_proj(feature)

            attn = torch.bmm(self.query_proj(y_hat), K.transpose(1, 2)) / self.scale  # [B, H, H_total]
            weight = F.softmax(attn, dim=-1)
            corr = torch.bmm(weight, V)  # [B, H, D]
            weather_scores.append(corr)

        correction = torch.stack(weather_scores, dim=-1).sum(dim=-1)  # [B, H, D]
        combined = self.score_proj(correction)
        P = F.softmax(torch.sum(combined * y_hat, dim=-1, keepdim=True) / self.scale, dim=1)
        final = torch.sum(P * y_hat, dim=1, keepdim=True)  # [B, 1, D]
        return y_hat + final.expand_as(y_hat)


class AutoformerForecast(nn.Module):
    def __init__(self, in_channels=1, d_model=64, num_factors=5, top_k=3, kernel_size=25, forecast_horizon=24):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.d_model = d_model

        self.input_proj = nn.Linear(in_channels, d_model)

        self.encoder = nn.ModuleList([
            AutoformerBlock(d_model, kernel_size, top_k),
            AutoformerBlock(d_model, kernel_size, top_k)
        ])

        self.trend_decoder = DecoderBlock(d_model, d_model, forecast_horizon)
        self.seasonal_decoder = DecoderBlock(d_model, d_model, forecast_horizon)

        self.mfa = MultiFactorAttention(d_model, num_factors)

    def forward(self, x, weather_hist, weather_fore):
        B, C, T = x.shape  # [B, C=1, T]
        x = self.input_proj(x.permute(0, 2, 1)).permute(0, 2, 1)  # project to [B, d_model, T]

        seasonal, trend, _ = self.encoder[0](x)
        seasonal, trend, seasonal2 = self.encoder[1](seasonal + trend)

        trend_out = self.trend_decoder(trend, self.forecast_horizon)        # [B, d_model, H]
        seasonal_out = self.seasonal_decoder(seasonal2, self.forecast_horizon)  # [B, d_model, H]

        decoded = trend_out + seasonal_out                                 # [B, d_model, H]
        decoded = decoded.permute(0, 2, 1)                                  # [B, H, d_model]

        corrected = self.mfa(decoded, weather_hist, weather_fore)          # [B, H, d_model]
        return corrected.permute(0, 2, 1)                                   # [B, d_model, H]
