import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from di_rnn import DIRNN
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# m - example: 6 × 48 = 3 days of recent history
# n - use 6 periodic values from the same time point in previous 6 days

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        se = (y_true - y_pred) ** 2
        return torch.sqrt(torch.mean(se) + self.eps)


def preprocess_data(csv_path, m=4, n=3, freq='30min', splits=(0.7, 0.1, 0.2)):
    print("📥 Loading and preprocessing data...")
    df = pd.read_csv(csv_path, sep=';', decimal=',')
    df['ts'] = pd.to_datetime(df['ts'])
    df = df.set_index('ts').sort_index()
    df = df.resample(freq).mean()
    df['vrednost'] = df['vrednost'].interpolate()

    total_len = len(df)
    train_end = int(total_len * splits[0])
    val_end = train_end + int(total_len * splits[1])

    df_train = df.iloc[:train_end].copy()
    df_val   = df.iloc[train_end:val_end].copy()
    df_test  = df.iloc[val_end:].copy()

    print("🔢 Scaling values...")
    scaler = MinMaxScaler()
    df_train.loc[:, 'scaled'] = scaler.fit_transform(df_train[['vrednost']])
    df_val.loc[:, 'scaled']   = scaler.transform(df_val[['vrednost']])
    df_test.loc[:, 'scaled']  = scaler.transform(df_test[['vrednost']])

    df_all = pd.concat([df_train, df_val, df_test])

    values = df_all['scaled']
    timestamps = values.index
    X_seq, X_per, y = [], [], []

    print("🧩 Constructing input sequences...")
    for idx in range(max(m, n * int(pd.Timedelta('1D') / pd.to_timedelta(freq))), len(timestamps)):
        t = timestamps[idx]
        s_start = t - m * pd.to_timedelta(freq)
        s_range = pd.date_range(start=s_start, periods=m, freq=freq)
        if not all(ts in df_all.index for ts in s_range):
            continue
        s_input = df_all.loc[s_range, 'scaled'].values

        p_input = []
        for i in range(1, n + 1):
            prev_day_time = t - pd.Timedelta(days=i)
            if prev_day_time in df_all.index:
                p_input.append(df_all.loc[prev_day_time, 'scaled'])
            else:
                break
        if len(p_input) != n:
            continue

        X_seq.append(s_input.reshape(m, 1))
        X_per.append(np.array(p_input).reshape(n, 1))
        y.append(values[t])

    X_seq = np.array(X_seq)
    X_per = np.array(X_per)
    y = np.array(y)

    total = len(y)
    train_end = int(total * splits[0])
    val_end = train_end + int(total * splits[1])

    print(f"✅ Dataset sizes → Train: {train_end}, Val: {val_end - train_end}, Test: {total - val_end}")

    data = {
        'train': (X_seq[:train_end], X_per[:train_end], y[:train_end]),
        'val':   (X_seq[train_end:val_end], X_per[train_end:val_end], y[train_end:val_end]),
        'test':  (X_seq[val_end:], X_per[val_end:], y[val_end:])
    }

    return data, scaler


def train_dirnn(model, train_data, val_data, epochs=20, lr_rnn=0.005, lr_bpnn=0.008, device='cpu'):
    print("🚂 Starting training...")
    model = model.to(device)
    optimizer = torch.optim.Adam([
    {'params': model.s_rnn.parameters(), 'lr': lr_rnn},
    {'params': model.p_rnn.parameters(), 'lr': lr_rnn},
    {'params': model.bpnn.parameters(), 'lr': lr_bpnn},
    ])

    criterion = RMSELoss()

    X_seq_train, X_per_train, y_train = [torch.tensor(x, dtype=torch.float32).to(device) for x in train_data]
    X_seq_val, X_per_val, y_val = [torch.tensor(x, dtype=torch.float32).to(device) for x in val_data]

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_seq_train, X_per_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_seq_val, X_per_val)
            val_loss = criterion(val_pred, y_val)

        print(f"\r📘 Epoch {epoch+1:02d}/{epochs} — \U0001F4CA Train Loss: {loss.item():.6f} | Val Loss: {val_loss.item():.6f}")


def evaluate_dirnn(model, test_data, scaler, device='cpu'):
    print("\n🧪 Evaluating model on test set...")
    model.eval()
    X_seq_test, X_per_test, y_test = [torch.tensor(x, dtype=torch.float32).to(device) for x in test_data]

    with torch.no_grad():
        preds = model(X_seq_test, X_per_test).cpu().numpy()

    y_true = y_test.cpu().numpy()
    y_pred = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    y_true_inv = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_true_inv, y_pred)
    mse = mean_squared_error(y_true_inv, y_pred)
    mape = mean_absolute_percentage_error(y_true_inv, y_pred)

    print("\n📊 Evaluation Metrics:")
    print(f"✅ MAE:  {mae:.4f}")
    print(f"✅ MSE:  {mse:.4f}")
    print(f"✅ MAPE: {mape:.4f}")


if __name__ == "__main__":
    print("🚀 Running DI-RNN pipeline...")
    data, scaler = preprocess_data('mm79158.csv', m=192, n=4, freq='30min')
    train_data = data['train']
    val_data = data['val']
    test_data = data['test']

    print("🧠 Initializing DIRNN model...")
    model = DIRNN(seq_input_size=1, per_input_size=1, hidden_size=12, bp_hidden_size=5, dropout=0.2)

    train_dirnn(model, train_data, val_data, epochs=20, lr_rnn=0.005, lr_bpnn=0.008, device='cpu')
    evaluate_dirnn(model, test_data, scaler, device='cpu')
