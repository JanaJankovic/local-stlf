from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import MeanAbsoluteError as MAEMetric
import pandas as pd
import os
import csv
import time
from data import prepare_data, prepare_sequences
from util import evaluate_forecasting_model


def build_cnn_lstm_model(input_shape, output_size=1):
    time_steps = input_shape[0]
    model = Sequential()

    model.add(Conv1D(48, 3, activation='relu', padding='same', input_shape=input_shape))
    if time_steps >= 4:
        model.add(MaxPooling1D(pool_size=2))
        time_steps //= 2

    model.add(Conv1D(32, 3, activation='relu', padding='same'))
    if time_steps >= 4:
        model.add(MaxPooling1D(pool_size=2))
        time_steps //= 2

    model.add(Conv1D(16, 3, activation='relu', padding='same'))
    if time_steps >= 4:
        model.add(MaxPooling1D(pool_size=2))
        time_steps //= 2

    model.add(Dropout(0.25))
    model.add(LSTM(20, return_sequences=True))
    model.add(LSTM(20, return_sequences=True))
    model.add(LSTM(20, return_sequences=False))
    model.add(Dropout(0.25))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(output_size))

    model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanAbsoluteError())
    return model


def train_model_core(model, X_train, y_train, X_val, y_val, scaler, model_path, log_dir, epoch_log_file, metrics_log_file, epochs=20):
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=MeanAbsoluteError(),
                  metrics=[MAEMetric(name='mae')])

    lr_schedule = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.8, min_lr=1e-5, verbose=1)
    checkpoint = ModelCheckpoint(model_path, monitor='val_mae', save_best_only=True, save_weights_only=False, verbose=1)

    os.makedirs(log_dir, exist_ok=True)
    if not os.path.exists(epoch_log_file):
        with open(epoch_log_file, 'w', newline='') as f:
            csv.writer(f).writerow(["start_epoch_time", "end_epoch_time", "epoch", "train_loss", "val_loss"])

    if not os.path.exists(metrics_log_file):
        with open(metrics_log_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'type', 'inference', 'MAE', 'MSE', 'RMSE', 'MAPE', 'R2', 'MDA', 'Spearman'])
            writer.writeheader()

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        history = model.fit(X_train, y_train,
                            validation_data=(X_val, y_val),
                            epochs=1,
                            batch_size=128,
                            callbacks=[lr_schedule, checkpoint],
                            verbose=0)
        end_time = time.strftime("%Y-%m-%d %H:%M:%S")
        train_loss = history.history['loss'][0]
        val_loss = history.history['val_loss'][0]

        with open(epoch_log_file, 'a', newline='') as f:
            csv.writer(f).writerow([start_time, end_time, epoch+1, train_loss, val_loss])

        for dtype, X, y in [('train', X_train, y_train), ('val', X_val, y_val)]:
            t0 = time.time()
            y_pred = model.predict(X, verbose=0)
            inference_time = time.time() - t0


            metrics = evaluate_forecasting_model(y, y_pred, scaler, inference_time, type=dtype)
            row = {
                'epoch': epoch + 1,
                'type': dtype,
                'inference': metrics['inference'],
                'MAE': metrics['MAE'],
                'MSE': metrics['MSE'],
                'RMSE': metrics['RMSE'],
                'MAPE': metrics['MAPE'],
                'R2': metrics['R2'],
                'MDA': metrics['MDA'],
                'Spearman': metrics['Spearman']
            }
            with open(metrics_log_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writerow(row)

    return model


def train_model(data, epochs=20):
    X_train, y_train, X_val, y_val, _, _, scaler = data

    print(f"\n🔁 Training with lookback={input_window}, horizon={forecast_horizon}")
    model = build_cnn_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        output_size=y_train.shape[1]
    )

    model_path = f"models/model_{input_window}_{forecast_horizon}.h5"
    epoch_log = f"logs/training_logs.csv"
    metrics_log = f"logs/training_eval.csv"

    model = train_model_core(model, X_train, y_train, X_val, y_val, scaler,
                             model_path=model_path,
                             log_dir='logs',
                             epoch_log_file=epoch_log,
                             metrics_log_file=metrics_log, 
                             epochs=epochs)



if __name__ == '__main__':
    df = prepare_data("data/mm79158.csv")
    input_window = 14 * 24
    forecast_horizon = 1

    X_train, y_train, X_val, y_val, X_test, y_test, scaler = prepare_sequences(
        df, input_window=input_window, forecast_horizon=forecast_horizon,
        val_ratio=0.1, test_ratio=0.3
    )

    data = (X_train, y_train, X_val, y_val, X_test, y_test, scaler)
    train_model(data, epochs=20)

