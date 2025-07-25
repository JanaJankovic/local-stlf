import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from model import swt_decompose
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- BUILD SEQUENCES ---
def build_sequences(data, lookback, horizon):
    X_raw, y_raw = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X_raw.append(data[i : i + lookback])
        y_raw.append(data[i + lookback : i + lookback + horizon])
    return np.array(X_raw), np.array(y_raw)


# --- SAFE SWT ---
def safe_swt(seq, wavelet, level):
    max_level = int(math.log2(len(seq)))
    return swt_decompose(seq, wavelet=wavelet, level=min(level, max_level))


# --- PREPARE DATA ---
def prepare_data(args):
    # --- LOAD ORIGINAL DATA ---
    df = pd.read_csv("data/mm79158.csv", parse_dates=["ts"])
    df.set_index("ts", inplace=True)
    df_hourly = df.resample("1h").mean().dropna()
    raw_data = df_hourly["vrednost"].values.reshape(-1)

    # --- SPLIT RAW SERIES FIRST ---
    total_len = len(raw_data)
    test_len = int(total_len * args["test"])
    val_len = int(total_len * args["val"])
    train_len = total_len - test_len - val_len

    raw_train = raw_data[:train_len]
    raw_val = raw_data[train_len : train_len + val_len]
    raw_test = raw_data[train_len + val_len :]

    # --- BUILD SEQUENCES PER SPLIT ---
    X_train, y_train = build_sequences(raw_train, args["lookback"], args["horizon"])
    X_val, y_val = build_sequences(raw_val, args["lookback"], args["horizon"])
    X_test, y_test = build_sequences(raw_test, args["lookback"], args["horizon"])

    # --- DECOMPOSE EACH SPLIT ---
    def swt_band_split(seqs):
        bands = None
        for seq in seqs:
            coeffs = safe_swt(seq, wavelet=args["wavelet"], level=args["level"])
            if bands is None:
                bands = [[] for _ in range(len(coeffs))]
            for j, (a, d) in enumerate(coeffs):
                if bands is None:
                    bands = [[] for _ in range(len(coeffs))]
                if j == 0:
                    bands[0].append(a)
                else:
                    bands[j].append(d)
        return [np.array(b).astype(np.float32) for b in bands]

    X_train_bands = swt_band_split(X_train)
    X_val_bands = swt_band_split(X_val)
    X_test_bands = swt_band_split(X_test)
    y_train_bands = swt_band_split(y_train)
    y_val_bands = swt_band_split(y_val)

    # --- SCALE PER BAND ---
    scalers_X = []
    scalers_y = []

    bands_for_scaling = [0, 1, 2]
    for i in bands_for_scaling:
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_bands[i] = scaler_X.fit_transform(
            X_train_bands[i].reshape(X_train_bands[i].shape[0], -1)
        ).reshape(X_train_bands[i].shape)
        X_val_bands[i] = scaler_X.transform(
            X_val_bands[i].reshape(X_val_bands[i].shape[0], -1)
        ).reshape(X_val_bands[i].shape)
        X_test_bands[i] = scaler_X.transform(
            X_test_bands[i].reshape(X_test_bands[i].shape[0], -1)
        ).reshape(X_test_bands[i].shape)

        y_train_bands[i] = scaler_y.fit_transform(
            y_train_bands[i].reshape(y_train_bands[i].shape[0], -1)
        ).reshape(y_train_bands[i].shape)
        y_val_bands[i] = scaler_y.transform(
            y_val_bands[i].reshape(y_val_bands[i].shape[0], -1)
        ).reshape(y_val_bands[i].shape)

        scalers_X.append(scaler_X)
        scalers_y.append(scaler_y)

    # --- TO TENSORS ---
    def to_tensors(bands):
        return [
            torch.tensor(b, dtype=torch.float32).unsqueeze(-1).to(DEVICE) for b in bands
        ]

    X_train_tensors = to_tensors(X_train_bands)
    X_val_tensors = to_tensors(X_val_bands)
    X_test_tensors = to_tensors(X_test_bands)
    y_train_tensors = to_tensors(y_train_bands)
    y_val_tensors = to_tensors(y_val_bands)

    # --- ALIGN TENSORS BY LENGTH ---
    n_train = X_train_tensors[0].size(0)
    n_val = X_val_tensors[0].size(0)
    n_test = X_test_tensors[0].size(0)

    assert all(
        t.size(0) == n_train for t in X_train_tensors + y_train_tensors
    ), "Mismatch in train set"
    assert all(
        t.size(0) == n_val for t in X_val_tensors + y_val_tensors
    ), "Mismatch in val set"
    assert all(t.size(0) == n_test for t in X_test_tensors), "Mismatch in test set"

    # --- DATASET ---
    train_dataset = TensorDataset(*X_train_tensors, *y_train_tensors)
    val_dataset = TensorDataset(*X_val_tensors, *y_val_tensors)
    test_dataset = TensorDataset(
        *X_test_tensors, torch.zeros(n_test, args["horizon"], 1).to(DEVICE)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args["batch_size"], shuffle=False
    )
    val_loader = DataLoader(val_dataset, batch_size=args["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=False)

    return (
        train_loader,
        val_loader,
        test_loader,
        (y_train, y_val, y_test),
        scalers_X,
        scalers_y,
        X_train_tensors,
        X_val_tensors,
        X_test_tensors,
    )
