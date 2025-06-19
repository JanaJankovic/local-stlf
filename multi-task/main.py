import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from model import MultiTaskOKL  # your custom model
import os
import csv
import torch.nn.functional as F

LOGS_CSV = "logs/loss_log.csv"
MODEL_PATH = 'models/model.pth'

def log_loss_csv(epoch, train_loss, val_loss, task_ids):
    write_header = not os.path.exists(LOGS_CSV)
    with open(LOGS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["task_id", "epoch", "train_loss", "val_loss"])
        for task_id in task_ids:
            writer.writerow([task_id, epoch + 1, round(train_loss, 6), round(val_loss, 6)])

# === DATASET WITH TASK AWARENESS ===
class LoadDataset(Dataset):
    def __init__(self, df, horizon=24):
        self.horizon = horizon
        features = ["time", "day_of_year", "day_type", "temperature_2m", "relative_humidity_2m"]
        self.X = torch.tensor(df[features].values, dtype=torch.float32)
        y = df["vrednost"].values
        self.y = torch.tensor([
            y[i:i + horizon] for i in range(len(y) - horizon)
        ], dtype=torch.float32)
        self.task_ids = torch.tensor(df["task_id"].values[:len(self.y)], dtype=torch.long)
        self.X = self.X[:len(self.y)]  # truncate X to match y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.task_ids[idx]


# === PREPROCESSING AND TASK SPLITTING ===
def load_and_preprocess(consumption_path, weather_path):
    df = pd.read_csv(consumption_path, parse_dates=["ts"])
    weather = pd.read_csv(weather_path, parse_dates=["datetime"])

    # Align weather with consumption timestamps
    weather.rename(columns={"datetime": "ts"}, inplace=True)
    df = pd.merge_asof(df.sort_values("ts"), weather.sort_values("ts"), on="ts")

    # Time features
    df["time_slot"] = (df["ts"].dt.hour * 60 + df["ts"].dt.minute) // 180
    df = df.groupby([df["ts"].dt.date, "time_slot"]).mean(numeric_only=True).reset_index()
    df["time"] = df["time_slot"] * 3
    df["day_of_year"] = pd.to_datetime(df["ts"]).dt.dayofyear
    df["day_type"] = pd.to_datetime(df["ts"]).dt.weekday

    # Keep necessary columns
    df = df[["time", "day_of_year", "day_type", "vrednost", "temperature_2m", "relative_humidity_2m"]]

    # Simulate 2 tasks
    size = df.shape[0]
    df["task_id"] = 0
    df.loc[size // 2 :, "task_id"] = 1
    return df

# === TASK-AWARE TRAINING FUNCTION ===
def train_model(model, loader, lambda_reg, device, epoch):
    model.train()
    X_list, Y_list, T_list = [], [], []

    for X, y, task_ids in loader:
        X_list.append(X)
        Y_list.append(y)
        T_list.append(task_ids)

    X = torch.cat(X_list).to(device)       # [N, D]
    Y = torch.cat(Y_list).to(device)       # [N, H]
    T = torch.cat(T_list).to(device)       # [N]
    G = model.compute_shared_basis(X)      # [N, p]

    # Aggregate data by task
    num_tasks = model.num_tasks
    p, H = model.p, Y.size(1)
    B = torch.zeros(p, num_tasks, H, device=device)

    for j in range(num_tasks):
        mask = (T == j)
        G_j = G[mask]                       # [n_j, p]
        y_j = Y[mask]                       # [n_j, H]
        GTG = G_j.T @ G_j + lambda_reg * torch.eye(p, device=device)
        GTy = G_j.T @ y_j
        B[:, j] = torch.linalg.solve(GTG, GTy)  # [p, H]

    model.B = B                            # Save weights

    # Solve for L (output kernel): L = (B^T B)^{1/2}
    B_flat = B.view(p, -1)                 # [p, T*H]
    BBT = B_flat.T @ B_flat                # [T*H, T*H]
    L = torch.linalg.norm(BBT, ord='nuc')  # Use nuclear norm as surrogate
    model.L = torch.eye(num_tasks, device=device) * (L / num_tasks)

    # Compute training error
    preds = model.predict_with_basis(G, T)  # [N, H]
    mse = F.mse_loss(preds, Y)
    print(f"[Epoch {epoch+1}] Closed-form MSE: {mse.item():.4f}")
    return mse.item()

# === TASK-AWARE EVALUATION FUNCTION ===
def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_true, all_tasks = [], [], []

    with torch.no_grad():
        for X, y, task_ids in loader:
            X, task_ids = X.to(device), task_ids.to(device)
            preds = model(X)[range(X.size(0)), task_ids]
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(y.numpy())
            all_tasks.extend(task_ids.cpu().numpy())

    # Convert lists of arrays to actual arrays
    all_preds = np.stack(all_preds)  # shape: [N, horizon]
    all_true = np.stack(all_true)    # shape: [N, horizon]
    all_tasks = np.array(all_tasks)  # shape: [N]

    # Group-wise MAPE and MNAE
    mape_list, mnae_list = [], []
    for task_id in np.unique(all_tasks):
        mask = all_tasks == task_id
        y_true = all_true[mask].reshape(-1)
        y_pred = all_preds[mask].reshape(-1)
        
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-3))) * 100
        mnae = np.mean(np.abs(y_true - y_pred)) / np.mean(np.abs(y_true))
        
        mape_list.append(mape)
        mnae_list.append(mnae)

    return mape_list, mnae_list


def split_by_task(df, val_ratio=0.2, test_ratio=0.2):
    train_parts, val_parts, test_parts = [], [], []

    for task_id in df["task_id"].unique():
        df_task = df[df["task_id"] == task_id]
        total_len = len(df_task)

        test_size = int(total_len * test_ratio)
        val_size = int(total_len * val_ratio)
        train_size = total_len - test_size - val_size

        train_parts.append(df_task.iloc[:train_size])
        val_parts.append(df_task.iloc[train_size:train_size + val_size])
        test_parts.append(df_task.iloc[train_size + val_size:])

    return (
        pd.concat(train_parts).reset_index(drop=True),
        pd.concat(val_parts).reset_index(drop=True),
        pd.concat(test_parts).reset_index(drop=True),
    )

# === MAIN PIPELINE ===
def run_training_pipeline(csv_path, weather_path, val_ratio=0.2, test_ratio=0.2, max_epochs=100, patience=10):
    if os.path.exists(LOGS_CSV):
        os.remove(LOGS_CSV)

    df = load_and_preprocess(csv_path, weather_path)
    train_df, val_df, test_df = split_by_task(df, val_ratio, test_ratio)

    train_data = LoadDataset(train_df)
    val_data = LoadDataset(val_df)
    test_data = LoadDataset(test_df)

    train_loader = DataLoader(train_data, batch_size=512, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=1024)
    test_loader = DataLoader(test_data, batch_size=1024)

    X_train = torch.tensor(train_df[["time", "day_of_year", "day_type"]].values, dtype=torch.float32)
    num_tasks = df["task_id"].nunique()
    model = MultiTaskOKL(X_train=X_train, num_tasks=num_tasks, p=10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    lambda_reg = 1e-3  # Regularization strength

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        avg_loss = train_model(model, train_loader, optimizer, loss_fn, lambda_reg, device, epoch)

        # Compute validation loss (not metrics)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y, task_ids in val_loader:
                X, y, task_ids = X.to(device), y.to(device), task_ids.to(device)
                preds = model(X)[range(X.size(0)), task_ids]
                val_loss += loss_fn(preds, y).item()
        val_loss /= len(val_loader)
        
        log_loss_csv(epoch, avg_loss, val_loss, task_ids=range(num_tasks))

        if val_loss < best_val_loss - 1e-4:  # improvement threshold
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} — no improvement for {patience} epochs.")
                break

    torch.save(model, MODEL_PATH)
    final_mape, final_mnae = evaluate_model(model, test_loader, device)
    print("\nFinal Evaluation on Test Set:")
    for i, (mape, mnae) in enumerate(zip(final_mape, final_mnae)):
        print(f"  Task {i}: MAPE = {mape:.2f}%, MNAE = {mnae:.4f}")


if __name__ == "__main__":
    run_training_pipeline("data/mm79158.csv", "data/slovenia_hourly_weather.csv", val_ratio=0.1, test_ratio=0.3, max_epochs=100, patience=15)
