import model 
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from model import SwtForecastingModel, swt_decompose, swt_reconstruct, EarlyStopping
import argparse
import matplotlib.pyplot as plt
import os
import csv
import ctypes
import math
from transformers import get_linear_schedule_with_warmup

# --- DEVICE SETUP ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DEVICE= torch.device("cpu")  # Force CPU for compatibility
log_path = "logs/train_log.csv"
print(f"🚀 Using device: {DEVICE}")

# --- CONFIGURATION ---
args = {
    "test": 0.3,       # test/val split ratio
    "val": 0.1, 
    "s": 8,        # target sequence length
    "w": 16,        # input sequence window
    "level": 3,     # user-defined max SWT level
    "wavelet": "db2"
}

# --- BUILD SEQUENCES ---
def build_sequences(data, w, s):
    X_raw, y_raw = [], []
    for i in range(len(data) - w - s + 1):
        X_raw.append(data[i:i + w])
        y_raw.append(data[i + w:i + w + s])
    return np.array(X_raw), np.array(y_raw)

# --- SAFE SWT ---
def safe_swt(seq, wavelet, level):
    max_level = int(math.log2(len(seq)))
    return swt_decompose(seq, wavelet=wavelet, level=min(level, max_level))

# --- PREPARE DATA ---
def prepare_data(args):
    # --- LOAD ORIGINAL DATA ---
    df = pd.read_csv("mm79158.csv", parse_dates=['ts'])
    df.set_index('ts', inplace=True)
    raw_data = df['vrednost'].values.reshape(-1)

    # --- SPLIT RAW SERIES FIRST ---
    total_len = len(raw_data)
    test_len = int(total_len * args['test'])
    val_len = int(total_len * args['val'])
    train_len = total_len - test_len - val_len

    raw_train = raw_data[:train_len]
    raw_val = raw_data[train_len:train_len + val_len]
    raw_test = raw_data[train_len + val_len:]

    # --- BUILD SEQUENCES PER SPLIT ---
    X_train, y_train = build_sequences(raw_train, args['w'], args['s'])
    X_val, y_val = build_sequences(raw_val, args['w'], args['s'])
    X_test, y_test = build_sequences(raw_test, args['w'], args['s'])

    # --- DECOMPOSE EACH SPLIT ---
    def swt_band_split(seqs):
        bands = None
        for seq in seqs:
            coeffs = safe_swt(seq, wavelet=args['wavelet'], level=args['level'])
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

    bands_for_scaling = [0]  # Only scale the first band
    for i in bands_for_scaling:
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_bands[i] = scaler_X.fit_transform(X_train_bands[i].reshape(X_train_bands[i].shape[0], -1)).reshape(X_train_bands[i].shape)
        X_val_bands[i] = scaler_X.transform(X_val_bands[i].reshape(X_val_bands[i].shape[0], -1)).reshape(X_val_bands[i].shape)
        X_test_bands[i] = scaler_X.transform(X_test_bands[i].reshape(X_test_bands[i].shape[0], -1)).reshape(X_test_bands[i].shape)

        y_train_bands[i] = scaler_y.fit_transform(y_train_bands[i].reshape(y_train_bands[i].shape[0], -1)).reshape(y_train_bands[i].shape)
        y_val_bands[i] = scaler_y.transform(y_val_bands[i].reshape(y_val_bands[i].shape[0], -1)).reshape(y_val_bands[i].shape)

        scalers_X.append(scaler_X)
        scalers_y.append(scaler_y)


    # --- TO TENSORS ---
    def to_tensors(bands):
        return [torch.tensor(b, dtype=torch.float32).unsqueeze(-1).to(DEVICE) for b in bands]

    X_train_tensors = to_tensors(X_train_bands)
    X_val_tensors = to_tensors(X_val_bands)
    X_test_tensors = to_tensors(X_test_bands)
    y_train_tensors = to_tensors(y_train_bands)
    y_val_tensors = to_tensors(y_val_bands)

    # --- ALIGN TENSORS BY LENGTH ---
    n_train = X_train_tensors[0].size(0)
    n_val = X_val_tensors[0].size(0)
    n_test = X_test_tensors[0].size(0)

    assert all(t.size(0) == n_train for t in X_train_tensors + y_train_tensors), "Mismatch in train set"
    assert all(t.size(0) == n_val for t in X_val_tensors + y_val_tensors), "Mismatch in val set"
    assert all(t.size(0) == n_test for t in X_test_tensors), "Mismatch in test set"

    # --- DATASET ---
    train_dataset = TensorDataset(*X_train_tensors, *y_train_tensors)
    val_dataset = TensorDataset(*X_val_tensors, *y_val_tensors)
    test_dataset = TensorDataset(*X_test_tensors, torch.zeros(n_test, args['s'], 1).to(DEVICE))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader, y_test, scalers_X, scalers_y, X_train_tensors, X_val_tensors, X_test_tensors


def train_model(train_loader, val_loader, X_train_tensors, args):
    import csv
    from transformers import get_linear_schedule_with_warmup

    model = SwtForecastingModel(
        input_size=1,
        time2vec_k=8,
        #d_model=480, # Original paper uses 480, but it is too large for this dataset
        d_model=64,
        #n_heads=12, # Original paper uses 12, but it is too large for this dataset
        n_heads=2,
        #d_ff=128, # Original paper uses 128, but it is too small for this dataset
        d_ff=128,
        n_enc_layers=2,
        n_dec_layers=0,
        forecast_steps=args['s'],
        output_bands=len(X_train_tensors)
    ).to(DEVICE)

    epochs = 100
    # Paper uses RMSprop, but it has given unpromising results and is quite slow.
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    loss_fn = torch.nn.MSELoss()
    #loss_fn = torch.nn.SmoothL1Loss() # experimental test
    early_stopper = EarlyStopping(patience=5, min_delta=1e-4, min_epochs=10)

    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss'])

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        num_batches = len(train_loader)
        print(f"\nEpoch {epoch+1:02d}/{epochs}")

        for batch_idx, batch in enumerate(train_loader):
            num_bands = len(batch) // 2
            x_band_tensor = [x.to(DEVICE) for x in batch[:num_bands]]
            y_band_tensor = [y.to(DEVICE) for y in batch[num_bands:]]

            x_stack = torch.stack(x_band_tensor, dim=1)  # [B, bands, W, 1]
            y_stack = torch.stack(y_band_tensor, dim=1)  # [B, bands, s, 1]

            preds = model(x_stack)
            loss = loss_fn(preds, y_stack)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            #scheduler.step()

            train_loss += loss.item() * y_stack.size(0)

            print(f"\r - Batch {batch_idx+1}/{num_batches} - loss: {loss.item():.6f}", end='', flush=True)

        # Print gradient norms at end of epoch
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"{name} grad norm: {param.grad.norm()}")

        train_loss /= len(train_loader.dataset)

        # === Validation ===
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                num_bands = len(batch) // 2
                x_band_tensor = [x.to(DEVICE) for x in batch[:num_bands]]
                y_band_tensor = [y.to(DEVICE) for y in batch[num_bands:]]

                x_stack = torch.stack(x_band_tensor, dim=1)
                y_stack = torch.stack(y_band_tensor, dim=1)

                preds = model(x_stack)
                loss = loss_fn(preds, y_stack)
                val_loss += loss.item() * y_stack.size(0)

        val_loss /= len(val_loader.dataset)
        print(f"\n[Epoch {epoch+1:02d}] Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f}")

        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, val_loss])

        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print("⏹️ Early stopping triggered.")
            model.load_state_dict(early_stopper.best_state_dict)
            break

    torch.save(model, "models/model.pth")
    print("✅ Model saved.")



if __name__ == "__main__":

    # Prevent sleep
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        # --- CONFIG ---

    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss'])  # header

    train_loader, val_loader, test_loader, y_test, scalers_X, scalers_y, X_train_tensors, X_val_tensors, X_test_tensors = prepare_data(args)
    train_model(train_loader, val_loader, X_train_tensors, args)
    print("Training complete. Model saved as 'models/model.pth'.")

    # Allow sleep again
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)