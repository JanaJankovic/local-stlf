import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from model import MultiTaskOKL  # replace with actual import
import os

LOG_FILE = "training_log.txt"

def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

# === DATA LOADING & PREPROCESSING ===
class LoadDataset(Dataset):
    def __init__(self, df):
        self.X = torch.tensor(df[["time", "day_of_year", "day_type"]].values, dtype=torch.float32)
        self.y = torch.tensor(df["value"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_and_preprocess(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])

    df["time_slot"] = (df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute) // 180
    df = df.groupby([df["timestamp"].dt.date, "time_slot"]).mean(numeric_only=True).reset_index()
    df["time"] = df["time_slot"] * 3
    df["day_of_year"] = pd.to_datetime(df["timestamp"]).dt.dayofyear
    df["day_type"] = pd.to_datetime(df["timestamp"]).dt.weekday

    df = df.drop(columns=["time_slot", "timestamp"], errors='ignore')
    return df

# === TRAINING FUNCTION ===
def train_model(model, loader, optimizer, loss_fn, device, epoch):
    model.train()
    total_loss = 0
    for i, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(X).squeeze()
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss
        log(f"[Epoch {epoch+1}] Batch {i+1}/{len(loader)} - Loss: {batch_loss:.4f}")
    return total_loss / len(loader)

# === EVALUATION FUNCTION ===
def evaluate_model(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            preds = model(X).squeeze().cpu().numpy()
            y_true.extend(y.numpy())
            y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-3))) * 100
    mnae = np.mean(np.abs(y_true - y_pred)) / np.mean(np.abs(y_true))
    return mape, mnae

# === MAIN ENTRYPOINT ===
def run_training_pipeline(csv_path):
    # Clear previous logs
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    df = load_and_preprocess(csv_path)
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

    train_data = LoadDataset(train_df)
    test_data = LoadDataset(test_df)

    train_loader = DataLoader(train_data, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1024)

    X_train = torch.tensor(train_df[["time", "day_of_year", "day_type"]].values, dtype=torch.float32)
    model = MultiTaskOKL(X_train=X_train, num_tasks=1, p=10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    log("=== Starting Training ===")
    for epoch in range(10):
        avg_loss = train_model(model, train_loader, optimizer, loss_fn, device, epoch)
        mape, mnae = evaluate_model(model, test_loader, device)
        log(f"Epoch {epoch+1} Summary: Avg Train Loss = {avg_loss:.4f} | MAPE = {mape:.2f}% | MNAE = {mnae:.4f}")

    log("\n=== Final Evaluation on Test Set ===")
    final_mape, final_mnae = evaluate_model(model, test_loader, device)
    log(f"Final MAPE: {final_mape:.2f}%")
    log(f"Final MNAE: {final_mnae:.4f}")

if __name__ == "__main__":
    run_training_pipeline("mm79158.csv")
