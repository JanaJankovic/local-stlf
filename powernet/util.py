import os
import csv
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from scipy.stats import spearmanr


def calculate_metrics(y_true, y_pred, elapsed_time, type='test'):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mda = np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred)))
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


import os
import csv
import numpy as np

def log_training_metrics(model_name, epoch, elapsed_time, y_true, y_pred, scaler, type='val', log_path='logs/training_eval.csv'):
    # Convert to numpy arrays if needed
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Reshape for inverse_transform (only if 1D or single-output)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    # Inverse transform to original scale
    y_true_inv = scaler.inverse_transform(y_true)
    y_pred_inv = scaler.inverse_transform(y_pred)

    # Calculate metrics
    metrics = calculate_metrics(y_true_inv, y_pred_inv, elapsed_time, type=type)

    # Prepare CSV row
    row = [model_name, epoch] + [metrics[k] for k in ['type', 'inference', 'MAE', 'MSE', 'RMSE', 'MAPE', 'R2', 'MDA', 'Spearman']]

    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    write_header = not os.path.exists(log_path)

    # Write to CSV
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['model', 'epoch', 'type', 'inference', 'MAE', 'MSE', 'RMSE', 'MAPE', 'R2', 'MDA', 'Spearman'])
        writer.writerow(row)



def log_loss_values(model_name, epoch, start_time, end_time, train_loss, val_loss, log_path='logs/training_log.csv'):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    write_header = not os.path.exists(log_path)
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['model', 'start_epoch_time', 'end_epoch_time', 'epoch', 'train_loss', 'val_loss'])
        writer.writerow([model_name, start_time, end_time, epoch, train_loss, val_loss])
