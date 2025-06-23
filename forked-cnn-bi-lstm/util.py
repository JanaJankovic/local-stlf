import numpy as np
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error, r2_score
)
from scipy.stats import spearmanr
import os
import time
import numpy as np
import pandas as pd
from keras.callbacks import Callback


def split_dataset(data, split_ratios=(0.7, 0.2, 0.1), daily_step=24, start=0):
    assert sum(split_ratios) == 1.0, "Split ratios must sum to 1.0"
    data = data[start:]
    total = len(data)

    n_train = int(total * split_ratios[0])
    n_val = int(total * split_ratios[1])
    n_test = total - n_train - n_val

    train = data[:n_train]
    val = data[n_train:n_train + n_val]
    test = data[n_train + n_val:]

    return train, val, test

# convert history into inputs (actual) and outputs (predicted) for training
def convert_train_val(data, n_input, n_out=24):
    # Ensure input is a NumPy array
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    X, y = [], []
    total_len = len(data)

    for i in range(total_len - n_input - n_out + 1):
        x_seq = data[i:i + n_input, 0].reshape(n_input, 1)
        y_seq = data[i + n_input:i + n_input + n_out, 0]
        X.append(x_seq)
        y.append(y_seq)

    return np.array(X), np.array(y)


# make a forecast
def forecast(model, history, n_input):
	# flatten data
	data = np.array(history)
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
	# retrieve last observations for input data
	input_x = data[-n_input:, 0]
	# reshape into [1, n_input, 1] that is 1 sample, sample data size, one feature
	input_x = input_x.reshape((1, len(input_x), 1))
	# forecast the next hour
	yhat = model.predict(input_x, verbose=0)
	# we only want the vector forecast
	yhat = yhat[0]
	return yhat


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


class MetricsLogger(Callback):
    def __init__(self, train_data, val_data, scaler, eval_log_path, loss_log_path, model_save_dir):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.scaler = scaler
        self.eval_log_path = eval_log_path
        self.loss_log_path = loss_log_path
        self.model_save_dir = model_save_dir
        os.makedirs(os.path.dirname(loss_log_path), exist_ok=True)
        os.makedirs(os.path.dirname(eval_log_path), exist_ok=True)
        os.makedirs(model_save_dir, exist_ok=True)

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        end_time = time.time()
        train_x, train_y = self.train_data
        val_x, val_y = self.val_data

        # Predict on both sets
        y_train_pred = self.model.predict(train_x, verbose=0)
        y_val_pred = self.model.predict(val_x, verbose=0)

        # Inverse scale
        y_train_true = self.scaler.inverse_transform(train_y.reshape(-1, 1)).flatten()
        y_train_pred = self.scaler.inverse_transform(y_train_pred.reshape(-1, 1)).flatten()
        y_val_true = self.scaler.inverse_transform(val_y.reshape(-1, 1)).flatten()
        y_val_pred = self.scaler.inverse_transform(y_val_pred.reshape(-1, 1)).flatten()

        train_metrics = calculate_metrics(y_train_true, y_train_pred, elapsed_time=(end_time - self.start_time), type='train')
        val_metrics = calculate_metrics(y_val_true, y_val_pred, elapsed_time=(end_time - self.start_time), type='val')

        # Log evaluation metrics
        for metrics in [train_metrics, val_metrics]:
            metrics['epoch'] = epoch + 1
            pd.DataFrame([metrics]).to_csv(self.eval_log_path, mode='a', header=not os.path.exists(self.eval_log_path), index=False)

        # Log losses + time
        entry = {
            "start_epoch_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time)),
            "end_epoch_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)),
            "epoch": epoch + 1,
            "train_loss": logs.get("loss", None),
            "val_loss": logs.get("val_loss", None)
        }
        pd.DataFrame([entry]).to_csv(self.loss_log_path, mode='a', header=not os.path.exists(self.loss_log_path), index=False)

        # Save model
        self.model.save(os.path.join(self.model_save_dir, f"model_epoch_{epoch+1}.h5"))
