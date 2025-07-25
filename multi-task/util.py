from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from scipy.stats import spearmanr
import numpy as np
import os
import csv
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

LOGS_PATH = "logs/training_log.csv"
METRICS_PATH = "logs/training_eval.csv"


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = (
        np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100
    )
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "R2": r2}


def fit_scalers_per_task(y, task_ids):
    """
    Returns a dict: {task_id: scaler}, and an array of scaled y.
    """
    scalers = {}
    y_scaled = np.zeros_like(y, dtype=np.float32)
    for task in np.unique(task_ids):
        idx = task_ids == task
        scaler = MinMaxScaler()
        y_task = y[idx].reshape(-1, 1)
        scaler.fit(y_task)
        y_scaled[idx] = scaler.transform(y_task).flatten()
        scalers[task] = scaler
    return scalers, y_scaled


def inverse_transform_per_task(y_scaled, task_ids, scalers):
    y_inv = np.zeros_like(y_scaled, dtype=np.float32)
    for task in np.unique(task_ids):
        idx = task_ids == task
        y_inv[idx] = (
            scalers[task].inverse_transform(y_scaled[idx].reshape(-1, 1)).flatten()
        )
    return y_inv


def log_loss_csv(epoch, start_time, end_time, train_loss, val_loss, task_ids):
    with open(LOGS_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        for task_id in task_ids:
            writer.writerow(
                [
                    task_id,
                    epoch + 1,
                    start_time,
                    end_time,
                    round(train_loss, 6),
                    round(val_loss, 6),
                ]
            )


def log_metrics_csv(epoch, model, loader, device, eval_type="test"):
    _, _, metrics = evaluate_model(model, loader, device, eval_type)
    with open(METRICS_PATH, "a", newline="") as f:
        writer = csv.writer(f)

        for task_id, m in metrics.items():
            writer.writerow(
                [
                    task_id,
                    epoch + 1,
                    m["type"],
                    f"{m['inference']:.6f}",
                    f"{m['MAE']:.6f}",
                    f"{m['MSE']:.6f}",
                    f"{m['RMSE']:.6f}",
                    f"{m['MAPE']:.2f}",
                    f"{m['R2']:.4f}",
                    f"{m['MDA']:.4f}",
                    f"{m['Spearman']:.4f}",
                ]
            )


def evaluate_model(model, loader, device, eval_type):
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
        mask = task_ids_all == task_id
        X_task = X_all[mask]
        y_task = y_all[mask]

        start_time = time.time()
        G_task = model.compute_shared_basis(X_task)
        preds_task = model.predict_with_basis(G_task, task_id.repeat(len(X_task)))
        elapsed_time = time.time() - start_time

        y_true_np = y_task.cpu().numpy()
        y_pred_np = preds_task.cpu().numpy()

        y_true_dict[task_id_int] = y_true_np
        y_pred_dict[task_id_int] = y_pred_np

        y_true_flat = y_true_np.flatten()
        y_pred_flat = y_pred_np.flatten()

        metrics = calculate_metrics(
            y_true_flat, y_pred_flat, elapsed_time=elapsed_time, type=eval_type
        )
        metrics_by_task[task_id_int] = metrics

    return y_true_dict, y_pred_dict, metrics_by_task


def save_predictions_per_task(
    y_true, y_pred, task_ids, out_path="logs/prediction_data.csv"
):
    data = []
    for task in np.unique(task_ids):
        indices = np.where(task_ids == task)[0]
        for i, idx in enumerate(indices):
            data.append([task, i, y_true[idx], y_pred[idx]])
    df = pd.DataFrame(data, columns=["task_id", "idx_in_task", "y_true", "y_pred"])
    df.to_csv(out_path, index=False)
