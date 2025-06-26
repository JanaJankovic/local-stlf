import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import csv
from model import MultiTaskOKL
from data import load_and_preprocess, LoadDataset, split_by_task
from util import calculate_metrics, evaluate_model, log_loss_csv, log_metrics_csv
import time

# === Config ===
FORECAST_LEN = 1
EPOCHS = 20
BATCH_SIZE = 64
LOGS_PATH = 'logs/training_log.csv'
METRICS_PATH = 'logs/training_eval.csv'
MODEL_PATH = 'models/'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, loader, lambda_reg, device, epoch, alternate_L=True):
    model.train()
    X_list, Y_list, T_list = [], [], []

    print(f"\n[Epoch {epoch+1}] Training started")

    for i, (X, y, task_ids) in enumerate(loader):
        print(f"  [Batch {i+1}/{len(loader)}] X: {X.shape}, y: {y.shape}, tasks: {task_ids.unique().tolist()}")
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


def run_training_pipeline(csv_path, val_ratio=0.2, test_ratio=0.2, max_epochs=100, patience=10):
    df = load_and_preprocess(csv_path)
    train_df, val_df, test_df, scalers = split_by_task(df, val_ratio, test_ratio)
    
    train_data = LoadDataset(train_df, horizon=FORECAST_LEN)
    val_data = LoadDataset(val_df, horizon=FORECAST_LEN)
    test_data = LoadDataset(test_df, horizon=FORECAST_LEN)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    # 🧠 Use scaled X_train
    task0_scaler = scalers[0]["feature_scaler"]
    X_train_scaled = task0_scaler.transform(train_df[["time", "day_of_year", "day_type"]])
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)

    num_tasks = df["task_id"].nunique()
    model = MultiTaskOKL(X_train=X_train, num_tasks=num_tasks, horizon=FORECAST_LEN, p=100)

    model.to(DEVICE)

    lambda_reg = 1e-3
    # best_val_loss = float("inf")
    # epochs_no_improve = 0

    for epoch in range(max_epochs):
        print(f"\n🚀 Starting Epoch {epoch+1}/{max_epochs}")
        start_time = time.time()
        avg_loss = train_model(model, train_loader, lambda_reg, DEVICE, epoch, alternate_L=True)
        end_time = time.time()

        print(f"✅ Finished training epoch {epoch+1} in {end_time - start_time:.2f}s. Avg loss: {avg_loss:.6f}")
        print(f"🔍 Evaluating on validation set...")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (X, y, task_ids) in enumerate(val_loader):
                print(f"  [Val Batch {i+1}/{len(val_loader)}]")
                X, y, task_ids = X.to(DEVICE), y.to(DEVICE), task_ids.to(DEVICE)
                G = model.compute_shared_basis(X)
                preds = model.predict_with_basis(G, task_ids)
                val_loss += F.mse_loss(preds, y).item()
        val_loss /= len(val_loader)

        print(f"📉 Validation loss for epoch {epoch+1}: {val_loss:.6f}")
        print(f"📋 Logging training/validation loss to CSV...")

        log_loss_csv(
            epoch, 
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)), 
            avg_loss, 
            val_loss, 
            task_ids=range(num_tasks)
        )

        log_metrics_csv(epoch, model, train_loader, DEVICE, scalers, 'train')
        log_metrics_csv(epoch, model, val_loader, DEVICE, scalers, 'val')

        print(f"💾 Saving model for epoch {epoch+1}...")
        torch.save(model, f"{MODEL_PATH}/model_epoch_{epoch+1}.pt")

        # if val_loss < best_val_loss - 1e-4:
        #     best_val_loss = val_loss
        #     epochs_no_improve = 0
        # else:
        #     epochs_no_improve += 1
        #     if epochs_no_improve >= patience:
        #         print(f"Early stopping at epoch {epoch+1} — no improvement for {patience} epochs.")
        #         break

if __name__ == "__main__":
    with open(LOGS_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["task_id", "epoch", "epoch_start_time", "epoch_end_time", "train_loss", "val_loss"])


    with open(METRICS_PATH, "w", newline="") as f:
        csv.writer(f).writerow(['task_id', 'epoch', 'type', 'inference', 'MAE', 'MSE', 'RMSE', 'MAPE', 'R2', 'MDA', 'Spearman'])

    run_training_pipeline("data/merged.csv", val_ratio=0.1, test_ratio=0.3, max_epochs=EPOCHS, patience=EPOCHS)
