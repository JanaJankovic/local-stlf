import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F


# --- Eq (2): Series decomposition (moving average smoothing & seasonal extraction) ---
class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        # f_Avgpool: moving average for smoothing (trend extraction)
        self.avg_pool = nn.AvgPool1d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )

    def forward(self, x):
        trend = self.avg_pool(x)  # X_t = f_Avgpool(X) [trend component]
        if trend.shape[-1] != x.shape[-1]:
            trend = F.pad(
                trend, (0, x.shape[-1] - trend.shape[-1])
            )  # f_pad for length matching
        seasonal = x - trend  # X_s = X - X_t [seasonal component]
        return seasonal, trend


# --- Eq (3-5): Auto-Correlation mechanism with FFT (Section 3.2) ---
class AutoCorrelation(nn.Module):
    def __init__(self, d_model, top_k=3):
        super().__init__()
        self.top_k = top_k
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.final_proj = nn.Linear(d_model * top_k, d_model)  # aggregation

    def forward(self, query, key, value):
        B, C, T = query.size()
        # Linear projections (to get Q, K, V from input)
        q = self.query_proj(query.permute(0, 2, 1)).permute(0, 2, 1)
        k = self.key_proj(key.permute(0, 2, 1)).permute(0, 2, 1)
        v = self.value_proj(value.permute(0, 2, 1)).permute(0, 2, 1)

        # --- Eq (5): FFT-based auto-correlation calculation ---
        fft_q = torch.fft.rfft(q, dim=-1)
        fft_k = torch.fft.rfft(k, dim=-1)
        auto_corr = (
            torch.fft.irfft(fft_q * torch.conj(fft_k), n=T, dim=-1) / T
        )  # R_xx(tau)

        # --- Eq (4): Top-k autocorrelated lags and attention weights ---
        top_k = min(self.top_k, T // 2)
        lags = torch.topk(auto_corr, top_k, dim=-1).indices  # top-k periodicities
        weights = F.softmax(auto_corr, dim=-1)  # attention weights

        # --- Eq (4): Time-delay aggregation and weighted sum ---
        outputs = []
        for i in range(top_k):
            shift = lags[..., i]  # [B, C]
            idx = torch.arange(T, device=query.device)
            # Roll/shift v by lag for each channel/batch (time-delay aggregation)
            rolled = torch.stack(
                [
                    torch.roll(v[b, c], int(shift[b, c]), dims=0)
                    for b in range(B)
                    for c in range(C)
                ]
            )
            rolled = rolled.view(B, C, T)
            weight = (
                weights.gather(-1, shift.unsqueeze(-1)).squeeze(-1).unsqueeze(-1)
            )  # softmax weights for top-k lags
            outputs.append(rolled * weight)  # weighted aggregation

        agg = torch.cat(outputs, dim=1)  # [B, C*top_k, T]
        agg = self.final_proj(agg.permute(0, 2, 1)).permute(
            0, 2, 1
        )  # final projection back to d_model
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


# --- Eq (6): Scoring mechanism for multi-factor attention (cross-attention) ---
class ScoringMechanism(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.scale = d_model**0.5
        self.proj = nn.Linear(
            1, d_model
        )  # Project weather factor to match embedding dimension

    def forward(self, y_hat, weather_seq):
        # y_hat: [B, T, D], weather_seq: [B, T_weather, F]
        B, T, D = y_hat.shape
        scores = []
        for i in range(weather_seq.shape[-1]):
            p = weather_seq[:, :, i : i + 1]  # [B, T_weather, 1]
            p_proj = self.proj(p)  # [B, T_weather, D]
            # --- Eq (6): S_p→x = Softmax((p x^T) / sqrt(d_model)) ---
            S = (
                torch.bmm(y_hat, p_proj.transpose(1, 2)) / self.scale
            )  # [B, T, T_weather]
            S = F.softmax(S, dim=-1)
            scores.append(S)
        return scores


# --- Eq (7): Correction mechanism using multi-factor scores and NWP ---
class CorrectionMechanism(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.scale = d_model**0.5

    def forward(self, scores, weather_seq, y_hat):
        # scores: list of [B, T, T_weather], weather_seq: [B, T_weather, F], y_hat: [B, T, D]
        B, T, D = y_hat.shape
        num_features = len(scores)
        weather_seq = weather_seq.permute(0, 2, 1)  # [B, F, T_weather]

        # For each factor, aggregate weather information weighted by attention scores
        weighted_sum = (
            torch.stack(
                [
                    torch.bmm(
                        scores[i], weather_seq[:, i : i + 1, :].transpose(1, 2)
                    ).squeeze(-1)
                    for i in range(num_features)
                ],
                dim=-1,
            )
            .sum(dim=-1)
            .unsqueeze(-1)
            .expand(-1, -1, D)
        )

        # --- Eq (7): Combine with forecasted load y_hat ---
        dot = (
            torch.sum(weighted_sum * y_hat, dim=-1, keepdim=True) / self.scale
        )  # [B, T, 1]
        P = F.softmax(dot, dim=1)  # Correction weights
        corrected = y_hat + y_hat * P
        return corrected


# --- Fig. 4 ---
class AutoformerForecast(nn.Module):
    def __init__(
        self,
        d_model,
        kernel_size=25,
        top_k=3,
        horizon=24,
        num_encoder_layers=2,
        num_decoder_layers=1,
    ):
        super().__init__()
        self.d_model = d_model
        self.horizon = horizon
        self.input_proj = nn.Linear(1, d_model)

        # Encoder and decoder stacks (each AutoformerBlock includes decomposition+autocorr)
        self.encoders = nn.ModuleList(
            [
                AutoformerBlock(d_model, kernel_size, top_k)
                for _ in range(num_encoder_layers)
            ]
        )
        self.final_decomp = SeriesDecomposition(kernel_size)
        self.decoders = nn.ModuleList(
            [
                AutoformerBlock(d_model, kernel_size, top_k)
                for _ in range(num_decoder_layers)
            ]
        )

        self.scoring = ScoringMechanism(d_model)
        self.corrector = CorrectionMechanism(d_model)

    def forward(self, load_series, weather_factors, nwp_forecast):
        # Project input to embedding
        x = self.input_proj(load_series.permute(0, 2, 1)).permute(0, 2, 1)

        # Encoder stack
        for encoder in self.encoders:
            x, _, _ = encoder(x)
        enc_out = x

        # Final decomposition: split into seasonal and trend
        seasonal_init, trend_init = self.final_decomp(enc_out)

        # Decoder stack: add trend_init after each decoder block
        seasonal_decoded = seasonal_init
        for decoder in self.decoders:
            seasonal_decoded, _, _ = decoder(seasonal_decoded)
            seasonal_decoded = seasonal_decoded + trend_init

        model_out = seasonal_decoded  # Already includes trend from all adds

        # Multi-factor AM: Scoring and correction
        weather_seq = torch.cat(
            [weather_factors, nwp_forecast], dim=1
        )  # [B, T_weather, F]
        scores = self.scoring(model_out.permute(0, 2, 1), weather_seq)
        corrected = self.corrector(scores, weather_seq, model_out.permute(0, 2, 1))

        return corrected[:, : self.horizon, 0]
