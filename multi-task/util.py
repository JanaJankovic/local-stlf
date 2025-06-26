from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error, r2_score
)
from scipy.stats import spearmanr
import numpy as np
import os
import csv
import torch

LOGS_PATH = 'logs/training_log.csv'
METRICS_PATH = 'logs/training_eval.csv'

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

def log_loss_csv(epoch, start_time, end_time, train_loss, val_loss, task_ids):
    with open(LOGS_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        for task_id in task_ids:
            writer.writerow([task_id, epoch + 1, start_time, end_time, round(train_loss, 6), round(val_loss, 6)])


def log_metrics_csv(epoch, model, loader, device, scalers, eval_type='test'):
    _, _, metrics = evaluate_model(model, loader, device, scalers, eval_type)
    with open(METRICS_PATH, 'a', newline='') as f:
        writer = csv.writer(f)

        for task_id, m in metrics.items():
            writer.writerow([
                task_id,
                epoch + 1,
                m['type'],
                f"{m['inference']:.6f}",
                f"{m['MAE']:.6f}",
                f"{m['MSE']:.6f}",
                f"{m['RMSE']:.6f}",
                f"{m['MAPE']:.2f}",
                f"{m['R2']:.4f}",
                f"{m['MDA']:.4f}",
                f"{m['Spearman']:.4f}"
            ])


def evaluate_model(model, loader, device, scalers, eval_type):
    import time
    model.eval()
    all_X, all_y, all_task_ids = [], [], []

    with torch.no_grad():
        for X, y, task_ids in loader:
            all_X.append(X.to(device))
            all_y.append(y.to(device))
            all_task_ids.append(task_ids.to(device))

    X_all = torch.cat(all_X)
    y_all = torch.cat(all_y)
    task_ids_all = torch.cat(all_task_ids)

    metrics_by_task = {}
    y_true_dict = {}
    y_pred_dict = {}

    for task_id in torch.unique(task_ids_all):
        task_id_int = int(task_id.item())
        mask = (task_ids_all == task_id)
        X_task = X_all[mask]
        y_task = y_all[mask]

        start_time = time.time()
        G_task = model.compute_shared_basis(X_task)
        preds_task = model.predict_with_basis(G_task, task_id.repeat(len(X_task)))
        elapsed_time = time.time() - start_time

        y_true_np = y_task.cpu().numpy()
        y_pred_np = preds_task.cpu().numpy()

        scaler = scalers[task_id_int]["target_scaler"]
        y_true_orig = scaler.inverse_transform(y_true_np)
        y_pred_orig = scaler.inverse_transform(y_pred_np)

        y_true_dict[task_id_int] = y_true_orig
        y_pred_dict[task_id_int] = y_pred_orig

        y_true_flat = y_true_orig.flatten()
        y_pred_flat = y_pred_orig.flatten()

        metrics = calculate_metrics(
            y_true_flat,
            y_pred_flat,
            elapsed_time=elapsed_time,
            type=eval_type
        )
        metrics_by_task[task_id_int] = metrics

    return y_true_dict, y_pred_dict, metrics_by_task
