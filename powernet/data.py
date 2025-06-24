
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch

def create_calendar_features(df, timestamp_col='ts'):
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df.set_index(timestamp_col, inplace=True)
    # Resample to hourly and interpolate
    df = df.resample('h').mean()
    df = df.interpolate(method='time', limit_direction='both')
    df.index.name = timestamp_col
    df['day_of_month'] = df.index.day
    df['day_of_week'] = df.index.weekday
    df['hour_of_day'] = df.index.hour
    df['period_of_day'] = df['hour_of_day'].apply(lambda h: 1 if 6 <= h < 18 else 0)
    df['is_weekend'] = df['day_of_week'].apply(lambda d: 1 if d >= 5 else 0)
    return df.reset_index()

def preprocess_weather_data(path, missing_threshold=0.15):
    df = pd.read_csv(path, parse_dates=['time'])
    df.set_index('time', inplace=True)
    # Ensure complete hourly range
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
    df = df.reindex(full_range)
    df.index.name = 'time'
    missing_ratio = df.isna().mean()
    df = df.drop(columns=missing_ratio[missing_ratio > missing_threshold].index.tolist())
    df = df.interpolate(method='time', limit_direction='both')
    return df

def join_calendar_and_weather(load_df, weather_df, timestamp_col='ts'):
    load_df[timestamp_col] = pd.to_datetime(load_df[timestamp_col])
    load_df = load_df.set_index(timestamp_col)
    merged_df = load_df.join(weather_df, how='inner')
    return merged_df.reset_index().rename(columns={'index': 'ts'})

def prepare_powernet_data(df, target_col='vrednost', timestamp_col='ts', test_size=0.2, val_size=0.1):
    df = df.drop(columns=[timestamp_col])
    y = df[[target_col]].copy()
    X = df.drop(columns=[target_col])
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_ratio, shuffle=False)

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    y_train_scaled = target_scaler.fit_transform(y_train)
    X_val_scaled = feature_scaler.transform(X_val)
    y_val_scaled = target_scaler.transform(y_val)
    X_test_scaled = feature_scaler.transform(X_test)
    y_test_scaled = target_scaler.transform(y_test)

    print(f"Total samples: {len(df)} | Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled, feature_scaler, target_scaler

def prepare_powernet_data(df, target_col='vrednost', timestamp_col='ts', test_size=0.2, val_size=0.1):
    df = df.drop(columns=[timestamp_col])
    y = df[[target_col]].copy()
    X = df.drop(columns=[target_col])
    # Let's assume you want to split by ratio, using manual indexing
    n = len(X)
    train_split = int(n * (1 - test_size - val_size))
    val_split = int(n * (1 - test_size))

    # Manual slicing
    X_train = X[:train_split]
    y_train = y[:train_split]

    X_val = X[train_split:val_split]
    y_val = y[train_split:val_split]

    X_test = X[val_split:]
    y_test = y[val_split:]

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    y_train_scaled = target_scaler.fit_transform(y_train)
    X_val_scaled = feature_scaler.transform(X_val)
    y_val_scaled = target_scaler.transform(y_val)
    X_test_scaled = feature_scaler.transform(X_test)
    y_test_scaled = target_scaler.transform(y_test)

    return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled, feature_scaler, target_scaler

def build_sequence_data(X, y, lookback, horizon):
    seq_x, meta_x, targets = [], [], []
    for i in range(lookback, len(X) - horizon + 1):
        seq_x.append(y[i - lookback:i])               # <- past load values for LSTM
        meta_x.append(X[i, :])                        # <- all other meta features at prediction time
        targets.append(y[i:i + horizon].flatten())    # <- target load values
    return (
        torch.tensor(seq_x, dtype=torch.float32),      # [B, lookback, 1]
        torch.tensor(meta_x, dtype=torch.float32),     # [B, meta]
        torch.tensor(targets, dtype=torch.float32)     # [B, horizon]
    )
