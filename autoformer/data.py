import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import torch
import os
import joblib  # For scaler serialization

def load_data(load_df, weather_df):
    # Round timestamps down to the hour
    load_df['hour'] = load_df['ts'].dt.floor('h')
    weather_df['hour'] = weather_df['datetime'].dt.floor('h')

    # Aggregate load: sum per hour
    hourly_load = load_df.groupby('hour')['vrednost'].sum().reset_index()
    hourly_load.rename(columns={'hour': 'datetime', 'vrednost': 'load_kWh'}, inplace=True)

    # Aggregate weather: mean per hour
    hourly_weather = weather_df.groupby('hour')[[
        'temperature_2m', 'relative_humidity_2m',
        'windspeed_10m', 'winddirection_10m', 'precipitation'
    ]].mean().reset_index()
    hourly_weather.rename(columns={'hour': 'datetime'}, inplace=True)

    # Merge on hourly datetime
    merged_df = pd.merge(hourly_load, hourly_weather, on='datetime', how='inner')
    merged_df = merged_df.sort_values('datetime').reset_index(drop=True)

    print("Loaded and merged data by hourly aggregation.")
    return merged_df


def create_sequences(data, input_len=24, forecast_horizon=24):
    load_seq = []
    weather_hist_seq = []
    weather_fore_seq = []
    target_seq = []

    total_len = input_len + forecast_horizon

    for i in range(len(data) - total_len):
        input_block = data[i : i + input_len]
        forecast_block = data[i + input_len : i + total_len]

        load_seq.append(input_block[:, 0])
        weather_hist_seq.append(input_block[:, 1:])
        weather_fore_seq.append(forecast_block[:, 1:])
        target_seq.append(forecast_block[:, 0])

    print(f"Created {len(load_seq)} sequences of length {input_len} with horizon {forecast_horizon}.")
    return (
        np.array(load_seq),
        np.array(weather_hist_seq),
        np.array(weather_fore_seq),
        np.array(target_seq)
    )


def prepare_data(csv_path, seq_len, horizon, batch_size=32, val_size=0.1, test_size=0.3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs('obj', exist_ok=True)

    # Load and merge data
    print("Loading data...")
    df_load = pd.read_csv(csv_path, parse_dates=['ts'])
    df_weather = pd.read_csv('data/slovenia_hourly_weather.csv', parse_dates=['datetime'])
    df = load_data(df_load, df_weather)

    # Split before scaling
    n_total = len(df)
    test_split = int((1 - test_size) * n_total)
    val_split = int((1 - val_size) * test_split)

    df_train = df[:val_split]
    df_val = df[val_split:test_split]
    df_test = df[test_split:]

    # Fit scaler only on training data
    features = ['load_kWh', 'temperature_2m', 'relative_humidity_2m',
                'windspeed_10m', 'winddirection_10m', 'precipitation']
    scaler = MinMaxScaler()
    df_train_scaled = scaler.fit_transform(df_train[features])
    df_val_scaled = scaler.transform(df_val[features])
    df_test_scaled = scaler.transform(df_test[features])

    print("Data split and scaled.")
    print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    # Sequence creation
    X_train_l, X_train_w_hist, X_train_w_fore, y_train = create_sequences(df_train_scaled, seq_len, horizon)
    X_val_l, X_val_w_hist, X_val_w_fore, y_val = create_sequences(df_val_scaled, seq_len, horizon)
    X_test_l, X_test_w_hist, X_test_w_fore, y_test = create_sequences(df_test_scaled, seq_len, horizon)

    # Convert to tensor datasets
    train_ds = TensorDataset(torch.tensor(X_train_l).unsqueeze(1).float(),
                             torch.tensor(X_train_w_hist).float(),
                             torch.tensor(X_train_w_fore).float(),
                             torch.tensor(y_train).float())

    val_ds = TensorDataset(torch.tensor(X_val_l).unsqueeze(1).float(),
                           torch.tensor(X_val_w_hist).float(),
                           torch.tensor(X_val_w_fore).float(),
                           torch.tensor(y_val).float())

    test_ds = TensorDataset(torch.tensor(X_test_l).unsqueeze(1).float(),
                            torch.tensor(X_test_w_hist).float(),
                            torch.tensor(X_test_w_fore).float(),
                            torch.tensor(y_test).float())

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, pin_memory=True)

    return train_loader, val_loader, test_loader, scaler