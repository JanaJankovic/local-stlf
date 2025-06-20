from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error, r2_score
)
from scipy.stats import spearmanr
import numpy as np
import time


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


def evaluate_forecasting_model(y_pred, y_test, scaler, inference, type):
    # Inverse scale if scaler is given
    if scaler is not None:
        y_pred = scaler.inverse_transform(y_pred)
        y_test = scaler.inverse_transform(y_test)


    return calculate_metrics(y_test, y_pred, inference, type) 