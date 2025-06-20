import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_data(csv_path):
    # === Load and preprocess energy data ===
    df = pd.read_csv(csv_path, sep=';', decimal=',')
    df['ts'] = pd.to_datetime(df['ts'])
    df.set_index('ts', inplace=True)

    # Resample to 30-minute intervals and interpolate
    df_resampled = df.resample('1h').mean()
    df_resampled['vrednost'] = df_resampled['vrednost'].interpolate(method='time')

    # Extract time-based features
    df_resampled['hour_24'] = df_resampled.index.hour + 1  # 1 to 24
    df_resampled['day_of_week'] = df_resampled.index.dayofweek

    # === Load holidays and tag rows ===
    holidays = pd.read_csv("data/slovenian_holidays_2016_2018.csv")
    holidays['holiday_date'] = pd.to_datetime(holidays['holiday_date'])
    holiday_set = set(holidays['holiday_date'].dt.normalize())
    df_resampled['is_holiday'] = df_resampled.index.normalize().isin(holiday_set).astype(int)

    # === One-hot encode hour_48, day_of_week, and is_holiday ===
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(df_resampled[['hour_24', 'day_of_week', 'is_holiday']])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(['hour_24', 'day_of_week', 'is_holiday']),
        index=df_resampled.index
    )

    # === Combine with original value and sort
    df_final = pd.concat([df_resampled[['vrednost']], encoded_df], axis=1)
    df_final = df_final.sort_index(ascending=True)
    df = df_final.copy()
    return df

def prepare_sequences(df, input_window=48, forecast_horizon=1, val_ratio=0.1, test_ratio=0.1):
    """
    Prepares sliding window sequences for multistep forecasting and applies MinMax scaling to 'vrednost' safely.

    Parameters:
    - df: DataFrame with 'vrednost' as the first column and features after
    - input_window: number of timesteps in the input window
    - forecast_horizon: number of timesteps to predict
    - val_ratio: validation size ratio
    - test_ratio: test size ratio

    Returns:
    - X_train, y_train, X_val, y_val, X_test, y_test, vrednost_scaler
    """
    df = df.copy()  # avoid modifying original

    # === Step 1: Determine sizes
    total_len = len(df) - input_window - forecast_horizon + 1
    val_size = int(total_len * val_ratio)
    test_size = int(total_len * test_ratio)
    train_size = total_len - val_size - test_size

    # Find raw index range for safe fitting
    fit_end_idx = train_size + input_window

    # === Step 2: Fit MinMaxScaler on 'vrednost' using only training range
    # Fit scaler on training data only
    vrednost_scaler = MinMaxScaler()
    df.loc[:df.index[fit_end_idx - 1], 'vrednost'] = vrednost_scaler.fit_transform(
        df.loc[:df.index[fit_end_idx - 1], ['vrednost']]
    ).ravel()

    # Transform validation + test data
    df.loc[df.index[fit_end_idx:], 'vrednost'] = vrednost_scaler.transform(
        df.loc[df.index[fit_end_idx:], ['vrednost']]
    ).ravel()


    # === Step 4: Generate sequences
    values = df.values
    X, y = [], []
    for i in range(total_len):
        X.append(values[i:i + input_window])
        y.append(values[i + input_window:i + input_window + forecast_horizon, 0])  # target = 'vrednost'

    X = np.array(X)
    y = np.array(y)

    if forecast_horizon == 1:
        y = y.reshape(-1, 1)

    # === Step 5: Final chronological split
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    return X_train, y_train, X_val, y_val, X_test, y_test, vrednost_scaler