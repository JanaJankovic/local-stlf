import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import LoadIntervalForecaster
from sklearn.preprocessing import MinMaxScaler


def aggregate_and_join(load_df, weather_df):
    """
    Aggregates 15-minute load data to hourly average, then joins with hourly weather data.
    
    Assumes:
    - load_df has 'ts' (timestamp) and 'vrednost' (load value) columns
    - weather_df has 'datetime' (timestamp) and hourly weather features

    Returns:
    - Merged DataFrame with hourly load + weather data
    """

    # Step 1: Convert timestamps to datetime
    load_df = load_df.copy()
    weather_df = weather_df.copy()
    
    load_df['ts'] = pd.to_datetime(load_df['ts'])
    weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])

    # Step 2: Round down to hour and aggregate load
    load_df['hour'] = load_df['ts'].dt.floor('H')
    hourly_load = load_df.groupby('hour')['vrednost'].mean().reset_index()
    hourly_load.rename(columns={'hour': 'datetime', 'vrednost': 'load'}, inplace=True)

    # Step 3: Join with weather on 'datetime'
    merged = pd.merge(hourly_load, weather_df, on='datetime', how='inner')

    return merged[['vrednost', 'temperature_2m']]

def build_sequences(trend, temp, window=24):
    X, y = [], []
    for i in range(len(trend) - window):
        x = np.column_stack([trend[i:i+window], temp[i:i+window]])
        X.append(x)
        y.append(trend[i + window])
    return np.array(X), np.array(y)


if __name__ == "__main__":
    df_load = pd.read_csv('mm79158,csv')
    df_weather = pd.read_csv('slovenia_hourly_weather.csv')

    df = aggregate_and_join(df_load, df_weather)

    # === 1. Split raw input before SSA or scaling ===
    vrednost = df['vrednost'].values
    temperature = df['temperature_2m'].values

    n = len(vrednost)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    vrednost_train = vrednost[:train_end]
    vrednost_val = vrednost[train_end:val_end]
    vrednost_test = vrednost[val_end:]

    temp_train = temperature[:train_end]
    temp_val = temperature[train_end:val_end]
    temp_test = temperature[val_end:]

    # === 2. SSA on each split (only on training/val/test separately) ===
    forecaster = LoadIntervalForecaster(input_dim=2)

    trend_train, period_train, noise_train = forecaster.ssa_decompose(vrednost_train, window=48)
    trend_val, period_val, noise_val = forecaster.ssa_decompose(vrednost_val, window=48)
    trend_test, period_test, noise_test = forecaster.ssa_decompose(vrednost_test, window=48)

    # === 3. Fit MinMaxScaler only on train, apply to all ===
    trend_scaler = MinMaxScaler()
    temp_scaler = MinMaxScaler()

    trend_train_scaled = trend_scaler.fit_transform(trend_train.reshape(-1, 1)).flatten()
    trend_val_scaled = trend_scaler.transform(trend_val.reshape(-1, 1)).flatten()
    trend_test_scaled = trend_scaler.transform(trend_test.reshape(-1, 1)).flatten()

    temp_train_scaled = temp_scaler.fit_transform(temp_train.reshape(-1, 1)).flatten()
    temp_val_scaled = temp_scaler.transform(temp_val.reshape(-1, 1)).flatten()
    temp_test_scaled = temp_scaler.transform(temp_test.reshape(-1, 1)).flatten()

    # === 4. Compute DF values from noise (no scaling needed) ===

    # === 5. Build sequences (must cut tail to match DF size) ===

    # Align lengths (safeguard)

    #Forecast with interval

