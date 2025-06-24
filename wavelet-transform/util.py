from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error, r2_score
)
from scipy.stats import spearmanr
import numpy as np
import datetime
import torch
import pywt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_metrics(y_true, y_pred, elapsed_time, type='test'):
    # Flatten if needed
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # MDA - Mean Directional Accuracy
    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred)
    mda = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff))

    # Spearman correlation
    spearman_corr, _ = spearmanr(y_true, y_pred)

    return {
        'type': type,
        'inference': elapsed_time,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'MDA': mda,
        'Spearman': spearman_corr
    }


def swt_reconstruct_from_preds(pred_bands, wavelet='db2', scaler_y=None):
    aN = pred_bands[0]  # a2

    if scaler_y is not None:
        aN = scaler_y.inverse_transform(aN.reshape(1, -1)).flatten()

    coeffs = [(aN, d) for d in pred_bands[1:]]  # [(a2, d2), (a2, d1)]
    return pywt.iswt(coeffs, wavelet)

def evaluate_model(model, data_loader, y_true_array, scaler_y, forecast_steps, data_type='val', wavelet='db2'):
    model.eval()
    preds_scaled = []
    start_time = datetime.datetime.now()

    with torch.no_grad():
        for batch in data_loader:
            num_bands = len(batch) // 2
            x_bands = [x.to(DEVICE) for x in batch[:num_bands]]

            x_tensor = torch.stack(x_bands, dim=1)  # [B, bands, W, 1]
            out = model(x_tensor)                   # [B, bands, s, 1]
            out = out.detach().cpu().numpy().squeeze(-1)  # [B, bands, s]

            B, bands, s = out.shape
            for b in range(B):
                band_preds = out[b]  # [bands, s]
                recon = swt_reconstruct_from_preds(band_preds, wavelet=wavelet, scaler_y=scaler_y)
                preds_scaled.append(torch.tensor(recon[-s:], dtype=torch.float32))  # [s]

    preds_scaled = torch.stack(preds_scaled)  # [B, s]
    preds_unscaled = preds_scaled.view(-1).numpy()
    y_pred = preds_unscaled.reshape(-1, forecast_steps)[:, 0]  # [B]

    # ⚠ Use first column of the true original unscaled array
    y_true = y_true_array[:, 0]  # [B]

    elapsed = (datetime.datetime.now() - start_time).total_seconds()
    return y_pred, calculate_metrics(y_true, y_pred, elapsed_time=elapsed, type=data_type)


def autoregressive_forecast_strict(
    model, val_loader, scaler_y, y_test,
    lookback=24, forecast_steps=12, wavelet='db2', level=3
):
    # Load model
    model.eval()

    # Extract initial sequence (a3) from last validation batch
    val_batch = list(val_loader)[-1]
    a3_seq = val_batch[0][-1].squeeze(-1).cpu().numpy().flatten().tolist()  # only a3

    preds = []
    true_recon = []

    with torch.no_grad():
        for t in range(0, len(y_test) - forecast_steps + 1, forecast_steps):
            current_input = np.array(a3_seq[-lookback:])

            # Decompose with SWT
            coeffs = pywt.swt(current_input, wavelet=wavelet, level=level)
            input_bands = []
            for j, (a, d) in enumerate(coeffs):
                band = a if j == 0 else d
                band_tensor = torch.tensor(band, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                input_bands.append(band_tensor)

            x_tensor = torch.stack(input_bands, dim=1).to(DEVICE)  # [1, bands, W, 1]
            out = model(x_tensor).squeeze(0).squeeze(-1).cpu().numpy()  # [bands, forecast_steps]

            a3_pred = scaler_y.inverse_transform(out[0].reshape(1, -1)).flatten()
            detail_preds = out[1:]
            pred_coeffs = [(a3_pred, d) for d in detail_preds]

            recon_pred = pywt.iswt(pred_coeffs, wavelet=wavelet)
            preds.append(recon_pred[-forecast_steps:])

            # True target reconstruction
            true_seq = y_test[t:t + forecast_steps]
            if len(true_seq) < forecast_steps:
                break
            true_coeffs = pywt.swt(true_seq, wavelet=wavelet, level=len(pred_coeffs))
            true_recon_coeffs = [(true_coeffs[i][0], true_coeffs[i][1]) for i in range(len(pred_coeffs))]
            recon_true = pywt.iswt(true_recon_coeffs, wavelet=wavelet)
            true_recon.append(recon_true[-forecast_steps:])

            # Autoregressive update with predicted a3
            a3_seq.extend(a3_pred.tolist())

    # Evaluation
    return np.concatenate(preds)