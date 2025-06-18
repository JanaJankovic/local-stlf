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
    x_cond, x_pred, y = [], [], []

    feature_columns = df.columns.tolist()
    if 'vrednost' not in feature_columns:
        raise ValueError("'vrednost' column must be present as the target variable.")

    future_columns = [col for col in feature_columns if col != 'vrednost']

    for i in range(input_len, len(df) - forecast_len):
        cond_block = df.iloc[i - input_len:i]
        pred_block = df.iloc[i:i + forecast_len]
        target_block = pred_block['vrednost']

        x_cond.append(cond_block.values)
        x_pred.append(pred_block[future_columns].values)
        y.append(target_block.values)

    return np.array(x_cond), np.array(x_pred), np.array(y)


# === Dataset Split and Scaling ===
def split_and_scale_dataset(x_cond, x_pred, y, val_ratio=0.1, test_ratio=0.3, scale=True, batch_size=128):
    total = len(x_cond)
    test_size = int(total * test_ratio)
    val_size = int(total * val_ratio)
    train_size = total - test_size - val_size

    print(f"📊 Dataset split → Train: {train_size}, Val: {val_size}, Test: {test_size}")

    slices = {
        "train": slice(0, train_size),
        "val": slice(train_size, train_size + val_size),
        "test": slice(-test_size, None)
    }

    data = {
        "x_cond_train": x_cond[slices["train"]],
        "x_pred_train": x_pred[slices["train"]],
        "y_train": y[slices["train"]],
        "x_cond_val": x_cond[slices["val"]],
        "x_pred_val": x_pred[slices["val"]],
        "y_val": y[slices["val"]],
        "x_cond_test": x_cond[slices["test"]],
        "x_pred_test": x_pred[slices["test"]],
        "y_test": y[slices["test"]],
    }

    scaler_x_cond, scaler_x_pred, scaler_y = None, None, None
    if scale:
        print("🔃 Scaling features (x_cond, x_pred) and targets (y)...")
        n_features_cond = x_cond.shape[2]
        n_features_pred = x_pred.shape[2]

        scaler_x_cond = MinMaxScaler()
        scaler_x_pred = MinMaxScaler()
        scaler_y = MinMaxScaler()

        # Fit and transform x_cond
        data["x_cond_train"] = scaler_x_cond.fit_transform(
            data["x_cond_train"].reshape(-1, n_features_cond)
        ).reshape(data["x_cond_train"].shape)
        data["x_cond_val"] = scaler_x_cond.transform(
            data["x_cond_val"].reshape(-1, n_features_cond)
        ).reshape(data["x_cond_val"].shape)
        data["x_cond_test"] = scaler_x_cond.transform(
            data["x_cond_test"].reshape(-1, n_features_cond)
        ).reshape(data["x_cond_test"].shape)

        # Fit and transform x_pred
        data["x_pred_train"] = scaler_x_pred.fit_transform(
            data["x_pred_train"].reshape(-1, n_features_pred)
        ).reshape(data["x_pred_train"].shape)
        data["x_pred_val"] = scaler_x_pred.transform(
            data["x_pred_val"].reshape(-1, n_features_pred)
        ).reshape(data["x_pred_val"].shape)
        data["x_pred_test"] = scaler_x_pred.transform(
            data["x_pred_test"].reshape(-1, n_features_pred)
        ).reshape(data["x_pred_test"].shape)

        # Fit and transform y
        data["y_train"] = scaler_y.fit_transform(data["y_train"])
        data["y_val"] = scaler_y.transform(data["y_val"])
        data["y_test"] = scaler_y.transform(data["y_test"])


    def make_loader(xc, xp, y):
        return DataLoader(
            TensorDataset(
                torch.tensor(xc, dtype=torch.float32),
                torch.tensor(xp, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32)
            ),
            batch_size=batch_size,
            shuffle=False
        )

    return (
        make_loader(data["x_cond_train"], data["x_pred_train"], data["y_train"]),
        make_loader(data["x_cond_val"], data["x_pred_val"], data["y_val"]),
        make_loader(data["x_cond_test"], data["x_pred_test"], data["y_test"]),
        (scaler_x_cond, scaler_x_pred),
        scaler_y
    )



# === Final All-in-One Prep ===
def prepare_interformer_dataloaders_and_prediction(
    condition_df,
    input_len=336,
    forecast_len=48,
    val_ratio=0.1,
    test_ratio=0.3,
    batch_size=128,
    scale=True
):
    x_cond, x_pred, y = create_sliding_windows(condition_df, input_len=input_len, forecast_len=forecast_len)
    train_loader, val_loader, test_loader, scaler_x, scaler_y = split_and_scale_dataset(
        x_cond, x_pred, y, val_ratio, test_ratio, scale, batch_size
    )
  
    return train_loader, val_loader, test_loader, scaler_x, scaler_y
