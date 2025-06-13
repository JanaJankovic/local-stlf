import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

# === Utility ===
def calculate_dew_point(temp_c, rh):
    a, b = 17.27, 237.7
    alpha = ((a * temp_c) / (b + temp_c)) + np.log(rh / 100.0)
    return (b * alpha) / (a - alpha)

# === Step 1: Load and Resample Load Data ===
def load_and_resample_load(path):
    df = pd.read_csv(path, sep=';', decimal=',')
    df['ts'] = pd.to_datetime(df['ts'])
    df.set_index('ts', inplace=True)
    df = df.resample('1h').mean()
    df['vrednost'] = df['vrednost'].interpolate(method='time')
    return df

# === Step 2: Load Weather Data and Add Dew Point ===
def load_and_prepare_weather(path):
    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df = df.resample('1h').mean()
    df['dew_point'] = calculate_dew_point(df['temperature_2m'], df['relative_humidity_2m'])
    return df

# === Step 3: Merge Load and Weather ===
def merge_load_weather(load_df, weather_df):
    merged = pd.merge(load_df, weather_df, left_index=True, right_index=True, how='inner')
    merged = merged.dropna()  # Ensure no NaNs after merge
    return merged

# === Step 4: Generate Calendar Features ===
def generate_calendar_features(df, holiday_path):
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    holidays = pd.read_csv(holiday_path)
    holidays['holiday_date'] = pd.to_datetime(holidays['holiday_date'])
    holiday_set = set(holidays['holiday_date'].dt.normalize())
    df['is_holiday'] = df.index.normalize().isin(holiday_set).astype(int)
    return df

# === Step 5: One-hot Encode Categorical Features ===
def one_hot_encode_calendar(df):
    covariates = ['hour', 'day_of_week', 'is_weekend', 'is_holiday']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = encoder.fit_transform(df[covariates])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(covariates), index=df.index)
    return pd.concat([df.drop(columns=covariates), encoded_df], axis=1)

# === Final Step: Prepare All Historical Data ===
def preprocess_all(load_path, weather_path, holiday_path):
    load_df = load_and_resample_load(load_path)
    weather_df = load_and_prepare_weather(weather_path)
    merged_df = merge_load_weather(load_df, weather_df)
    calendar_df = generate_calendar_features(merged_df, holiday_path)
    return one_hot_encode_calendar(calendar_df)

# ✅ Fixed: Prepare Future Prediction Features Without 'vrednost'
def prepare_prediction_window(future_weather_path, holiday_path):
    weather_df = load_and_prepare_weather(future_weather_path)
    calendar_df = generate_calendar_features(weather_df, holiday_path)
    df = one_hot_encode_calendar(calendar_df)
    return df

# === Create Sliding Windows for Model ===
def create_sliding_windows(df, input_len=336, forecast_len=48):
    X, y = [], []
    for i in range(input_len, len(df) - forecast_len):
        X_window = df.iloc[i - input_len:i].values
        y_window = df.iloc[i:i + forecast_len]['vrednost'].values
        X.append(X_window)
        y.append(y_window)
    X = np.array(X)
    y = np.array(y)
    print(f"✔️ Sliding windows → X shape: {X.shape}, y shape: {y.shape}")
    return X, y

# === Dataset Split and Scaling ===
def split_and_scale_dataset(X, y, val_ratio=0.1, test_ratio=0.3, scale=True, batch_size=128):
    total = len(X)
    test_size = int(total * test_ratio)
    val_size = int(total * val_ratio)
    train_size = total - test_size - val_size

    print(f"📊 Dataset split → Train: {train_size}, Val: {val_size}, Test: {test_size}")

    X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[-test_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[-test_size:]

    scaler_x, scaler_y = None, None
    if scale:
        print("🔃 Scaling X (features) and y (forecast sequence)...")
        n_features = X.shape[2]

        # --- Scale X ---
        scaler_x = MinMaxScaler()
        X_train = scaler_x.fit_transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
        X_val   = scaler_x.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
        X_test  = scaler_x.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

        # --- Scale y (univariate forecast sequence) ---
        scaler_y = MinMaxScaler()
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
        y_val   = scaler_y.transform(y_val.reshape(-1, 1)).reshape(y_val.shape)
        y_test  = scaler_y.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val   = torch.tensor(X_val, dtype=torch.float32)
    X_test  = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val   = torch.tensor(y_val, dtype=torch.float32)
    y_test  = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size)
    val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
    test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    return train_loader, val_loader, test_loader, scaler_x, scaler_y


# === Prepare Prediction Input ===
def prepare_prediction_input(condition_df, prediction_df, input_len=336, forecast_len=48, scaler=None):
    # Fill missing target column temporarily if scaler expects it
    if 'vrednost' in condition_df.columns and 'vrednost' not in prediction_df.columns:
        prediction_df = prediction_df.copy()
        prediction_df['vrednost'] = 0.0  # dummy values for scaler

    # Align feature columns
    common_cols = condition_df.columns.intersection(prediction_df.columns)
    condition_df = condition_df[common_cols]
    prediction_df = prediction_df[common_cols]

    df_cond = condition_df.tail(input_len)
    df_pred = prediction_df.head(forecast_len)

    if df_cond.shape[0] < input_len:
        raise ValueError(f"Need {input_len} steps in condition, got {df_cond.shape[0]}")
    if df_pred.shape[0] < forecast_len:
        raise ValueError(f"Need {forecast_len} steps in prediction, got {df_pred.shape[0]}")

    combined = pd.concat([df_cond, df_pred])

    if combined.isnull().any().any():
        raise ValueError("NaNs detected after combining input slices.")

    if scaler is not None:
        scaled = scaler.transform(combined.values)

        # Drop dummy target if it was added
        n_features = combined.shape[1]
        if 'vrednost' in prediction_df.columns:
            idx_vrednost = combined.columns.get_loc('vrednost')
            scaled = np.delete(scaled, idx_vrednost, axis=1)
            n_features -= 1

        x_cond = scaled[:input_len].reshape(1, input_len, n_features)
        x_pred = scaled[input_len:].reshape(1, forecast_len, n_features)
    else:
        x_cond = df_cond.values.reshape(1, input_len, -1)
        x_pred = df_pred.values.reshape(1, forecast_len, -1)

    return torch.tensor(x_cond, dtype=torch.float32), torch.tensor(x_pred, dtype=torch.float32)

# === Final All-in-One Prep ===
def prepare_interformer_dataloaders_and_prediction(
    condition_df,
    prediction_df,
    input_len=336,
    forecast_len=48,
    val_ratio=0.1,
    test_ratio=0.3,
    batch_size=128,
    scale=True
):
    X, y = create_sliding_windows(condition_df, input_len=input_len, forecast_len=forecast_len)
    train_loader, val_loader, test_loader, scaler_x, scaler_y = split_and_scale_dataset(
        X, y, val_ratio, test_ratio, scale, batch_size
    )
    x_cond_pred, x_pred = prepare_prediction_input(condition_df, prediction_df, input_len, forecast_len, scaler_x)
    print(f"✅ Prepared all data | X_cond shape, X_pred shape: {x_cond_pred.shape, x_pred.shape}")
    return train_loader, val_loader, test_loader, x_pred, scaler_x, scaler_y
