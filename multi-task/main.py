import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import csv
from model import MultiTaskOKL
from sklearn.preprocessing import MinMaxScaler

LOGS_CSV = "logs/loss_log.csv"
MODEL_PATH = "models/model.pth"

def log_loss_csv(epoch, train_loss, val_loss, task_ids):
    write_header = not os.path.exists(LOGS_CSV)
    with open(LOGS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["task_id", "epoch", "train_loss", "val_loss"])
        for task_id in task_ids:
            writer.writerow([task_id, epoch + 1, round(train_loss, 6), round(val_loss, 6)])

class LoadDataset(Dataset):
    def __init__(self, df, horizon=24):
        self.horizon = horizon
        features = ["time", "day_of_year", "day_type"]
        self.X = torch.tensor(df[features].values, dtype=torch.float32)
        y = df["vrednost"].values
        self.y = torch.from_numpy(np.stack([y[i:i+horizon] for i in range(len(y)-horizon)])).float()
        self.task_ids = torch.tensor(df["task_id"].values[:len(self.y)], dtype=torch.long)
        self.X = self.X[:len(self.y)]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.task_ids[idx]

def load_and_preprocess(path):
    df = pd.read_csv(path, parse_dates=["ts"])
    df["time_slot"] = (df["ts"].dt.hour * 60 + df["ts"].dt.minute) // 180
    df = df.groupby([df["ts"].dt.date, "time_slot"]).mean(numeric_only=True).reset_index()
    df["time"] = df["time_slot"] * 3
    df["day_of_year"] = pd.to_datetime(df["ts"]).dt.dayofyear
    df["day_type"] = pd.to_datetime(df["ts"]).dt.weekday
    df = df[["time", "day_of_year", "day_type", "vrednost"]]
    size = df.shape[0]
    df["task_id"] = 0
    df.loc[size // 2:, "task_id"] = 1
    return df

def train_model(model, loader, lambda_reg, device, epoch, alternate_L=True):
    model.train()
    X_list, Y_list, T_list = [], [], []

    for X, y, task_ids in loader:
        X_list.append(X)
        Y_list.append(y)
        T_list.append(task_ids)

    X = torch.cat(X_list).to(device)
    Y = torch.cat(Y_list).to(device)
    T = torch.cat(T_list).to(device)

    print(f"\n=== Epoch {epoch+1} ===")
    print(f"X.shape: {X.shape}, Y.shape: {Y.shape}, T.shape: {T.shape}")

    num_tasks, p, H = model.num_tasks, model.p, Y.size(1)

    # Step 1: Build Kernel and compute A
    K = model.shared_basis.build_kernel(X, model.shared_basis.X_train)
    print(f"Kernel K shape: {K.shape}, mean: {K.mean().item():.4f}, std: {K.std().item():.4f}")

    T_full = torch.zeros(K.shape[0], num_tasks * H, device=K.device)
    for j in range(num_tasks):
        mask = (T == j)
        T_full[mask, j * H:(j + 1) * H] = Y[mask]
        print(f"  → Task {j} has {mask.sum().item()} samples")

    B_block = model.B.permute(1, 2, 0).reshape(num_tasks * H, p).to(device)
    YB = T_full @ B_block
    reg = lambda_reg * torch.eye(K.shape[1], device=K.device)

    A = torch.linalg.solve(K.T @ K + reg, K.T @ YB)
    model.shared_basis.A = A.to(device)
    G = K @ A

    print(f"A.shape: {A.shape}, G.shape: {G.shape}")

    # Step 2: Solve B
    B = torch.zeros(p, num_tasks, H, device=device)
    for j in range(num_tasks):
        mask = (T == j)
        G_j = G[mask]
        y_j = Y[mask]
        GTG = G_j.T @ G_j + lambda_reg * torch.eye(p, device=device)
        GTy = G_j.T @ y_j
        B[:, j] = torch.linalg.solve(GTG, GTy)
    model.B = B
    print(f"B.shape: {B.shape}, B std per task: {[B[:, j].std().item() for j in range(num_tasks)]}")

    # Step 3: Update L
    if alternate_L:
        B_concat = B.permute(1, 2, 0).reshape(num_tasks, -1)
        cov = B_concat @ B_concat.T
        eigvals, eigvecs = torch.linalg.eigh(cov)
        top_eigvecs = eigvecs[:, -p:]
        top_eigvals = eigvals[-p:]
        model.L = top_eigvecs @ torch.diag(top_eigvals) @ top_eigvecs.T
        print(f"Updated L with eigvals: {top_eigvals.tolist()}")

    preds = model.predict_with_basis(G, T)
    mse = F.mse_loss(preds, Y)
    print(f"MSE: {mse.item():.6f}")
    return mse.item()


def evaluate_model(model, loader, device, scalers):
    model.eval()
    all_preds, all_true, all_tasks = [], [], []

    with torch.no_grad():
        for X, y, task_ids in loader:
            X, y, task_ids = X.to(device), y.to(device), task_ids.to(device)
            G = model.compute_shared_basis(X)
            preds = model.predict_with_basis(G, task_ids)
            all_preds.append(preds.cpu())
            all_true.append(y.cpu())
            all_tasks.append(task_ids.cpu())

    all_preds = torch.cat(all_preds)
    all_true = torch.cat(all_true)
    all_tasks = torch.cat(all_tasks)

    mape_list, mnae_list = [], []
    for task_id in torch.unique(all_tasks):
        mask = (all_tasks == task_id)
        y_true = all_true[mask].numpy()
        y_pred = all_preds[mask].numpy()

        scaler = scalers[int(task_id.item())]["target_scaler"]
        y_true_orig = scaler.inverse_transform(y_true)
        y_pred_orig = scaler.inverse_transform(y_pred)

        y_true_flat = y_true_orig.reshape(-1)
        y_pred_flat = y_pred_orig.reshape(-1)

        mape = np.mean(np.abs((y_true_flat - y_pred_flat) / np.maximum(y_true_flat, 1e-3))) * 100
        mnae = np.mean(np.abs(y_true_flat - y_pred_flat)) / np.mean(np.abs(y_true_flat))

        mape_list.append(mape)
        mnae_list.append(mnae)

    return mape_list, mnae_list

def split_by_task(df, val_ratio=0.2, test_ratio=0.2):
    train_parts, val_parts, test_parts = [], [], []
    scalers = {}

    for task_id in df["task_id"].unique():
        df_task = df[df["task_id"] == task_id].copy()
        total_len = len(df_task)

        test_size = int(total_len * test_ratio)
        val_size = int(total_len * val_ratio)
        train_size = total_len - test_size - val_size

        train = df_task.iloc[:train_size].copy()
        val = df_task.iloc[train_size:train_size + val_size].copy()
        test = df_task.iloc[train_size + val_size:].copy()

        feature_cols = ["time", "day_of_year", "day_type"]
        feature_scaler = MinMaxScaler()
        feature_scaler.fit(train[feature_cols])

        target_scaler = MinMaxScaler()
        target_scaler.fit(train[["vrednost"]])

        for split in [train, val, test]:
            split[feature_cols] = feature_scaler.transform(split[feature_cols])
            split[["vrednost"]] = target_scaler.transform(split[["vrednost"]])

        scalers[task_id] = {
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
        }

        train_parts.append(train)
        val_parts.append(val)
        test_parts.append(test)

    return (
        pd.concat(train_parts).reset_index(drop=True),
        pd.concat(val_parts).reset_index(drop=True),
        pd.concat(test_parts).reset_index(drop=True),
        scalers
    )

def run_training_pipeline(csv_path, val_ratio=0.2, test_ratio=0.2, max_epochs=100, patience=10):
    if os.path.exists(LOGS_CSV):
        os.remove(LOGS_CSV)

    df = load_and_preprocess(csv_path)
    train_df, val_df, test_df, scalers = split_by_task(df, val_ratio, test_ratio)

    train_data = LoadDataset(train_df)
    val_data = LoadDataset(val_df)
    test_data = LoadDataset(test_df)

    train_loader = DataLoader(train_data, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1024)
    test_loader = DataLoader(test_data, batch_size=1024)

    # 🧠 Use scaled X_train
    task0_scaler = scalers[0]["feature_scaler"]
    X_train_scaled = task0_scaler.transform(train_df[["time", "day_of_year", "day_type"]])
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)

    num_tasks = df["task_id"].nunique()
    model = MultiTaskOKL(X_train=X_train, num_tasks=num_tasks, p=100)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    lambda_reg = 1e-5
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        avg_loss = train_model(model, train_loader, lambda_reg, device, epoch, alternate_L=True)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y, task_ids in val_loader:
                X, y, task_ids = X.to(device), y.to(device), task_ids.to(device)
                G = model.compute_shared_basis(X)
                preds = model.predict_with_basis(G, task_ids)
                val_loss += F.mse_loss(preds, y).item()
        val_loss /= len(val_loader)

        log_loss_csv(epoch, avg_loss, val_loss, task_ids=range(num_tasks))

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} — no improvement for {patience} epochs.")
                break

    torch.save(model, MODEL_PATH)
    final_mape, final_mnae = evaluate_model(model, test_loader, device, scalers)
    print("\nFinal Evaluation on Test Set:")
    for i, (mape, mnae) in enumerate(zip(final_mape, final_mnae)):
        print(f"  Task {i}: MAPE = {mape:.2f}%, MNAE = {mnae:.4f}")

if __name__ == "__main__":
    run_training_pipeline("data/mm79158.csv", val_ratio=0.1, test_ratio=0.1, max_epochs=100, patience=15)
