from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error, r2_score
)
from scipy.stats import spearmanr
import torch
import numpy as np
import os 
import csv
import time

def evaluate_model_on_loader(model, loader, device, scaler, target_index=0):
    model.eval()
    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for x_load, x_weather_hist, x_weather_fore, y in loader:
            x_load = x_load.to(device)
            x_weather_hist = x_weather_hist.to(device)
            x_weather_fore = x_weather_fore.to(device)
            y = y.to(device)

            output = model(x_load, x_weather_hist, x_weather_fore)

            y_true_all.append(y.cpu())
            y_pred_all.append(output.cpu())

    y_true = torch.cat(y_true_all).numpy()
    y_pred = torch.cat(y_pred_all).numpy()

    # Ensure y_true and y_pred are 2D
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    # === Inverse transform (only target column) ===
    dummy = np.zeros((y_true.shape[0], scaler.n_features_in_))
    dummy[:, target_index] = y_true[:, 0]
    y_true_inv = scaler.inverse_transform(dummy)[:, target_index]

    dummy[:, target_index] = y_pred[:, 0]
    y_pred_inv = scaler.inverse_transform(dummy)[:, target_index]

    return y_true_inv, y_pred_inv


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


def log_metrics_per_epoch(model, epoch, train_loader, val_loader, device, eval_log_path, scaler):
    os.makedirs(os.path.dirname(eval_log_path), exist_ok=True)

    fieldnames = ['epoch', 'type', 'inference', 'MAE', 'MSE', 'RMSE', 'MAPE', 'R2', 'MDA', 'Spearman']

    if epoch == 0 and not os.path.exists(eval_log_path):
        with open(eval_log_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    for dtype, loader in [('train', train_loader), ('val', val_loader)]:
        start_eval = time.perf_counter()
        y_true, y_pred = evaluate_model_on_loader(model, loader, device, scaler)
        elapsed = time.perf_counter() - start_eval
        metrics = calculate_metrics(y_true, y_pred, elapsed, type=dtype)
        metrics['epoch'] = epoch + 1

        ordered_metrics = {key: metrics[key] for key in fieldnames}

        with open(eval_log_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(ordered_metrics)
