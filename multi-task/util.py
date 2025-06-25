from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error, r2_score
)
from scipy.stats import spearmanr
import numpy as np
import os
import csv
import torch

LOGS_PATH = 'logs/train_log.csv'

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

def log_loss_csv(epoch, train_loss, val_loss, task_ids):
    write_header = not os.path.exists(LOGS_PATH)
    with open(LOGS_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["task_id", "epoch", "train_loss", "val_loss"])
        for task_id in task_ids:
            writer.writerow([task_id, epoch + 1, round(train_loss, 6), round(val_loss, 6)])


def evaluate_model(model, loader, device, scalers):
    model.eval()
    all_preds, all_true, all_tasks = [], [], []

    with torch.no_grad():
        for X, y, task_ids in loader:
            X, y, task_ids = X.to(device), y.to(device), task_ids.to(device)
            G = model.compute_shared_basis(X)
            preds = model.predict_with_basis(G, task_ids)
            all_preds.append(preds.cpu())
            all_true.append(y.cpu())
            all_tasks.append(task_ids.cpu())

    all_preds = torch.cat(all_preds)
    all_true = torch.cat(all_true)
    all_tasks = torch.cat(all_tasks)

    mape_list, mnae_list = [], []
    for task_id in torch.unique(all_tasks):
        mask = (all_tasks == task_id)
        y_true = all_true[mask].numpy()
        y_pred = all_preds[mask].numpy()

        scaler = scalers[int(task_id.item())]["target_scaler"]
        y_true_orig = scaler.inverse_transform(y_true)
        y_pred_orig = scaler.inverse_transform(y_pred)

        y_true_flat = y_true_orig.reshape(-1)
        y_pred_flat = y_pred_orig.reshape(-1)

        mape = np.mean(np.abs((y_true_flat - y_pred_flat) / np.maximum(y_true_flat, 1e-3))) * 100
        mnae = np.mean(np.abs(y_true_flat - y_pred_flat)) / np.mean(np.abs(y_true_flat))

        mape_list.append(mape)
        mnae_list.append(mnae)

    return mape_list, mnae_list