import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_resample(csv_path, freq):
    df = pd.read_csv(csv_path, sep=';', decimal=',')
    df['ts'] = pd.to_datetime(df['ts'])
    df.set_index('ts', inplace=True)
    df = df.sort_index().resample(freq).mean()
    df['vrednost'] = df['vrednost'].interpolate()
    return df

def split_dataframe(df, splits):
    total_len = len(df)
    train_end = int(total_len * splits[0])
    val_end = train_end + int(total_len * splits[1])
    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[val_end:].copy()
    )

def scale_data(df_train, df_val, df_test):
    scaler = MinMaxScaler()
    df_train['scaled'] = scaler.fit_transform(df_train[['vrednost']])
    df_val['scaled']   = scaler.transform(df_val[['vrednost']])
    df_test['scaled']  = scaler.transform(df_test[['vrednost']])
    return scaler, pd.concat([df_train, df_val, df_test])

def build_sequences(df_all, m, n, freq, horizon):
    X_seq, X_per, y = [], [], []
    values = df_all['scaled']
    timestamps = values.index
    min_required = max(m, n * int(pd.Timedelta('1D') / pd.to_timedelta(freq)))

    for idx in range(min_required, len(timestamps) - horizon):
        t = timestamps[idx]

        # Short-term input (S-RNN)
        s_start = t - m * pd.to_timedelta(freq)
        s_range = pd.date_range(start=s_start, periods=m, freq=freq)
        if not all(ts in df_all.index for ts in s_range):
            continue
        s_input = df_all.loc[s_range, 'scaled'].values.reshape(m, 1)

        # Periodic input (P-RNN)
        p_input = []
        for i in range(1, n + 1):
            prev_day_time = t - pd.Timedelta(days=i)
            if prev_day_time in df_all.index:
                p_input.append(df_all.loc[prev_day_time, 'scaled'])
            else:
                break
        if len(p_input) != n:
            continue
        p_input = np.array(p_input).reshape(n, 1)

        # Horizon target
        y_start = s_range[-1] + pd.to_timedelta(freq)
        y_range = pd.date_range(start=y_start, periods=horizon, freq=freq)
        if not all(ts in df_all.index for ts in y_range):
            continue
        y_output = df_all.loc[y_range, 'scaled'].values

        X_seq.append(s_input)
        X_per.append(p_input)
        y.append(y_output)

    return np.array(X_seq), np.array(X_per), np.array(y)

def final_split(X_seq, X_per, y, splits):
    total_samples = len(y)
    train_end = int(total_samples * splits[0])
    val_end = train_end + int(total_samples * splits[1])

    print(f"✅ Dataset sizes → Train: {train_end}, Val: {val_end - train_end}, Test: {total_samples - val_end}")

    return {
        'train': (X_seq[:train_end], X_per[:train_end], y[:train_end]),
        'val':   (X_seq[train_end:val_end], X_per[train_end:val_end], y[train_end:val_end]),
        'test':  (X_seq[val_end:], X_per[val_end:], y[val_end:])
    }

def preprocess_data(csv_path, m=4, n=3, freq='1h', splits=(0.6, 0.1, 0.3), horizon=1):
    print("📥 Loading and preprocessing data...")
    df = load_and_resample(csv_path, freq)
    df_train, df_val, df_test = split_dataframe(df, splits)
    print("🔢 Scaling values...")
    scaler, df_all = scale_data(df_train, df_val, df_test)
    print("🧩 Constructing input sequences...")
    X_seq, X_per, y = build_sequences(df_all, m, n, freq, horizon)
    data = final_split(X_seq, X_per, y, splits)
    return data, scaler, df_all
