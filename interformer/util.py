from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error, r2_score
)
from scipy.stats import spearmanr
import numpy as np
import torch
import time

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


def evaluate_model(model, data_loader, scaler_y, condition_df, eval_type, quantile_index=1,  device='cuda'):
    model.eval()

    y_true_all = []
    y_pred_all = []

    vred_index = list(condition_df.columns).index('vrednost')
    min_v = scaler_y.data_min_[vred_index]
    max_v = scaler_y.data_max_[vred_index]
    
    def invert_scaled(y):
        return y * (max_v - min_v) + min_v

    with torch.no_grad():
        start_time = time.time()
        for x_cond, x_pred, y in data_loader:
            x_cond, x_pred, y = x_cond.to(device), x_pred.to(device), y.to(device)

            # Predict
            preds, *_ = model(x_cond, x_pred)  # [B, Q, H]
            median_preds = preds[:, quantile_index, :]  # take median quantile

            # Move to CPU for metric eval
            y_true_all.append(y.cpu().numpy())
            y_pred_all.append(median_preds.cpu().numpy())
        end_time = time.time()

    y_true_all = np.concatenate(y_true_all, axis=0)
    y_pred_all = np.concatenate(y_pred_all, axis=0)

    # Invert scaling
    y_true_all = invert_scaled(y_true_all)
    y_pred_all = invert_scaled(y_pred_all)

    return y_true_all, y_pred_all, calculate_metrics(y_true_all, y_pred_all, end_time - start_time, eval_type)