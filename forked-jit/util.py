from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error, r2_score
)
from scipy.stats import spearmanr, ConstantInputWarning
import numpy as np
import torch
import joblib
import time
import warnings

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

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConstantInputWarning)
        spearman_corr, _ = spearmanr(y_true, y_pred)
        if np.isnan(spearman_corr):
            spearman_corr = 0.0  # or np.nan if you prefer


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


def predict(model, test_loader, device='cpu', scaler_dir='scalers'):
    scaler_target = joblib.load(f'{scaler_dir}/scaler_target.pkl')
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for enc_input, dec_input, _ in test_loader:
            enc_input = enc_input.to(device)
            dec_input = dec_input.to(device)
            output = model(enc_input, dec_input)
            all_predictions.append(output.cpu().numpy())

    y_pred_scaled = np.concatenate(all_predictions, axis=0)
    y_pred = scaler_target.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(y_pred_scaled.shape)
    return y_pred


def evaluate_model(model, data_loader, data_indices, df, data_type='test'):
    # --- Predict 
    start = time.time()
    y_pred = predict(model, data_loader, device=DEVICE, scaler_dir='scalers')
    end = time.time()

    y_true = df['SMA_7'].values
    y_true = y_true[data_indices]

    return y_true, y_pred, calculate_metrics(y_true, y_pred, elapsed_time=(end - start), type=data_type)
