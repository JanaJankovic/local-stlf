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
    def __init__(self, d_model, top_k=3):
        super().__init__()
        self.top_k = top_k
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.final_proj = nn.Linear(d_model * top_k, d_model)

    def forward(self, query, key, value):
        B, C, T = query.size()

        q = self.query_proj(query.permute(0, 2, 1)).permute(0, 2, 1)
        k = self.key_proj(key.permute(0, 2, 1)).permute(0, 2, 1)
        v = self.value_proj(value.permute(0, 2, 1)).permute(0, 2, 1)

        fft_q = torch.fft.rfft(q, dim=-1)
        fft_k = torch.fft.rfft(k, dim=-1)
        auto_corr = torch.fft.irfft(fft_q * torch.conj(fft_k), n=T, dim=-1) / T

        top_k = min(self.top_k, T // 2)
        lags = torch.topk(auto_corr, top_k, dim=-1).indices  # [B, C, k]
        weights = F.softmax(auto_corr, dim=-1)

        outputs = []
        for i in range(top_k):
            shift = lags[..., i]  # [B, C]
            idx = torch.arange(T, device=query.device)
            rolled = torch.stack([torch.roll(v[b, c], int(shift[b, c]), dims=0) for b in range(B) for c in range(C)])
            rolled = rolled.view(B, C, T)
            weight = weights.gather(-1, shift.unsqueeze(-1)).squeeze(-1).unsqueeze(-1)  # [B, C, 1]
            outputs.append(rolled * weight)

        agg = torch.cat(outputs, dim=1)  # [B, C*top_k, T]
        agg = self.final_proj(agg.permute(0, 2, 1)).permute(0, 2, 1)
        return agg

class AutoformerBlock(nn.Module):
    def __init__(self, d_model, kernel_size=25, top_k=3):
        super().__init__()
        self.decomp = SeriesDecomposition(kernel_size)
        self.auto_corr = AutoCorrelation(d_model, top_k)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x shape: [B, C, T]
        seasonal, trend = self.decomp(x)
        # Match the full signature: Q = K = V = seasonal
        seasonal_ac = self.auto_corr(seasonal, seasonal, seasonal)
        # Add residual connection after auto-corr
        seasonal = seasonal + seasonal_ac
        # Combine seasonal + trend
        combined = seasonal + trend
        # Project using Linear (permute for nn.Linear)
        combined = self.proj(combined.permute(0, 2, 1)).permute(0, 2, 1)
        return combined, trend, seasonal

class ScoringMechanism(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.scale = d_model ** 0.5
        self.proj = nn.Linear(1, d_model)

    def forward(self, y_hat, weather_seq):
        B, T, D = y_hat.shape
        scores = []

        for i in range(weather_seq.shape[-1]):
            p = weather_seq[:, :, i:i+1]  # [B, T_weather, 1]
            p_proj = self.proj(p)        # [B, T_weather, D]

            S = torch.bmm(y_hat, p_proj.transpose(1, 2)) / self.scale  # [B, T, T_weather]
            S = F.softmax(S, dim=-1)
            scores.append(S)

        return scores

class CorrectionMechanism(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.scale = d_model ** 0.5

    def forward(self, scores, weather_seq, y_hat):
        B, T, D = y_hat.shape
        num_features = len(scores)
        weather_seq = weather_seq.permute(0, 2, 1)  # [B, F, T_weather]

        weighted_sum = torch.stack([
            torch.bmm(scores[i], weather_seq[:, i:i+1, :].transpose(1, 2)).squeeze(-1)
            for i in range(num_features)
        ], dim=-1).sum(dim=-1).unsqueeze(-1).expand(-1, -1, D)

        dot = torch.sum(weighted_sum * y_hat, dim=-1, keepdim=True) / self.scale  # [B, T, 1]
        P = F.softmax(dot, dim=1)
        corrected = y_hat + y_hat * P
        return corrected

class AutoformerForecast(nn.Module):
    def __init__(self, d_model, kernel_size=25, top_k=3, horizon=24):
        super().__init__()
        self.d_model = d_model
        self.horizon = horizon
        self.input_proj = nn.Linear(1, d_model)

        self.encoder = AutoformerBlock(d_model, kernel_size, top_k)
        self.decoder_block1 = AutoformerBlock(d_model, kernel_size, top_k)
        self.decoder_block2 = AutoformerBlock(d_model, kernel_size, top_k)

        self.final_decomp = SeriesDecomposition(kernel_size)

        self.scoring = ScoringMechanism(d_model)
        self.corrector = CorrectionMechanism(d_model)

    def forward(self, load_series, weather_factors, nwp_forecast):
        x = self.input_proj(load_series.permute(0, 2, 1)).permute(0, 2, 1)
        enc_out, _, _ = self.encoder(x)
        seasonal_init, trend_init = self.final_decomp(enc_out)
        dec_out1, _, _ = self.decoder_block1(seasonal_init)
        dec_out2, _, _ = self.decoder_block2(dec_out1)
        dec_out = dec_out2 + trend_init

        weather_seq = torch.cat([weather_factors, nwp_forecast], dim=1)  # [B, T_weather, F]
        scores = self.scoring(dec_out.permute(0, 2, 1), weather_seq)
        corrected = self.corrector(scores, weather_seq, dec_out.permute(0, 2, 1))
        return corrected[:, :self.horizon, 0]



