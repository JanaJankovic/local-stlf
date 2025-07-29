import torch
import numpy as np
import pandas as pd
import os
import csv
import time
import random
from model import MultiTaskOKL
from data import load_and_preprocess, split_by_task
from util import (
    calculate_metrics,
    save_predictions_per_task,
    fit_scalers_per_task,
    inverse_transform_per_task,
)

# === Config ===
METRICS_PATH = "logs/training_eval.csv"
MODEL_PATH = "models/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_RANDOM_SEARCH = 20  # Number of random search iterations
BLOCK_SIZE = 5000
P = 200


def calculate_metrics_per_task(y_true, y_pred, task_ids, elapsed_time):
    unique_tasks = np.unique(task_ids)
    all_metrics = []
    metric_keys = ["MAE", "MSE", "RMSE", "MAPE", "R2"]
    for task in unique_tasks:
        idx = task_ids == task
        metrics = calculate_metrics(y_true[idx], y_pred[idx])
        metrics["task_id"] = task
        metrics["time"] = (
            elapsed_time  # Optional: if you want to keep it, but do NOT average this
        )
        all_metrics.append(metrics)
    # Macro-average: average each numeric metric across tasks
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in metric_keys}
    # Optional: add avg elapsed time if needed
    avg_metrics["time"] = elapsed_time
    return avg_metrics, all_metrics


def compute_KtK_and_KtYB_blockwise(X, YB, kernel_fn, device="cuda"):
    """
    Computes KᵀK and KᵀYB without instantiating full K.
    - X: [N, D]
    - YB: [N, P]
    Returns:
        - KtK: [N, N]
        - KtYB: [N, P]
    """
    N = X.size(0)
    P = YB.size(1)
    KtK = torch.zeros(N, N, device=device)
    KtYB = torch.zeros(N, P, device=device)

    block_size = BLOCK_SIZE
    total_blocks = (N + block_size - 1) // block_size
    print(f"Starting blockwise computation: {total_blocks} x {total_blocks} blocks")

    for i in range(0, N, block_size):
        i_end = min(i + block_size, N)
        X_i = X[i:i_end].to(device)
        YB_i = YB[i:i_end].to(device)

        for j in range(0, N, block_size):
            j_end = min(j + block_size, N)
            X_j = X[j:j_end].to(device)

            K_ij = kernel_fn(X_i, X_j)  # shape [i_len, j_len]

            # Compute partial blocks
            KtK[j:j_end, j:j_end] += K_ij.T @ K_ij
            KtYB[j:j_end] += K_ij.T @ YB_i

            print(f"Computed block ({i}:{i_end}, {j}:{j_end})")

    print("Symmetrizing KᵀK for stability...")
    KtK = (KtK + KtK.T) / 2

    print("Blockwise computation finished.")
    return KtK, KtYB


def fit_multitask_model_lowrank(X, y, task_ids, num_tasks, params):
    model = MultiTaskOKL(
        X_train=X,
        num_tasks=num_tasks,
        horizon=1,
        p=P,
        sigma_t=params["sigma_t"],
        sigma_d=params["sigma_d"],
    )
    model.to(DEVICE)

    if y.ndim == 1:
        y = y.unsqueeze(1)

    H = y.size(1)
    T = task_ids
    N = X.size(0)
    p = model.p

    # Step 1: Build task-wide matrix T_full ∈ [N, T×H]
    T_full = torch.zeros(N, num_tasks * H, device=DEVICE)
    for j in range(num_tasks):
        mask = T == j
        T_full[mask, j * H : (j + 1) * H] = y[mask]

    # Step 2: Initialize random B_block ∈ [T×H, p]
    B_block = torch.randn(num_tasks * H, p, device=DEVICE)

    # Step 3: Compute YB ∈ [N, p]
    YB = T_full @ B_block

    # Step 4: Compute K @ A = G in blocks
    print("🧠 Building feature map G = K @ A in blocks...")
    G = torch.zeros(N, p, device=DEVICE)
    block_size = 1024

    for i in range(0, N, block_size):
        i_end = min(i + block_size, N)
        K_block = model.shared_basis.build_kernel(
            X[i:i_end].to(DEVICE), X.to(DEVICE)
        )  # [i_end - i, N]
        G[i:i_end] = K_block @ YB  # G = K @ A where A ≈ YB

    # Step 5: Compute B using G
    print("📐 Solving for B matrix")
    B = torch.zeros(p, num_tasks, H, device=DEVICE)
    for j in range(num_tasks):
        mask = T == j
        G_j = G[mask]
        y_j = y[mask]
        GTG = G_j.T @ G_j + params["lambda_reg"] * torch.eye(p, device=DEVICE)
        GTy = G_j.T @ y_j
        B[:, j] = torch.linalg.solve(GTG, GTy)

    model.B = B
    model.shared_basis.A = YB  # So technically A = YB, implicitly

    print("✅ Low-rank multi-task model trained.")
    return model


def predict_multitask(model, X_pred, task_ids_pred):
    G_pred = model.shared_basis(X_pred)
    preds = model.predict_with_basis(G_pred, task_ids_pred)
    return preds


def random_search_training(
    csv_path,
    val_ratio=0.2,
    test_ratio=0.2,
    lambda_p=1,
    p=P,
    sigma_t_grid=[2.0, 4.0, 6.0],
    sigma_d_grid=[1.0, 2.0, 2.5, 3.5, 5.0],
    n_trials=N_RANDOM_SEARCH,
    device=DEVICE,
):

    Xdf, ydf = load_and_preprocess(
        csv_path,
    )
    X_train_df, y_train_df, X_val_df, y_val_df, X_test_df, y_test_df = split_by_task(
        Xdf, ydf, val_ratio, test_ratio
    )
    kernel_features = ["time", "day_of_year", "day_type"]

    train_task_ids = X_train_df["task_id"].values
    val_task_ids = X_val_df["task_id"].values
    test_task_ids = X_test_df["task_id"].values

    scalers, y_train_scaled = fit_scalers_per_task(y_train_df.values, train_task_ids)

    def apply_scalers(y, task_ids, scalers):
        y_scaled = np.zeros_like(y, dtype=np.float32)
        for task in np.unique(task_ids):
            idx = task_ids == task
            if task in scalers:
                y_scaled[idx] = scalers[task].transform(y[idx].reshape(-1, 1)).flatten()
            else:
                y_scaled[idx] = y[idx]
        return y_scaled

    y_val_scaled = apply_scalers(y_val_df.values, val_task_ids, scalers)
    y_test_scaled = apply_scalers(y_test_df.values, test_task_ids, scalers)

    def to_tensor(df):
        return torch.tensor(df.values, dtype=torch.float32)

    X_train = to_tensor(X_train_df[kernel_features])
    X_val = to_tensor(X_val_df[kernel_features])
    X_test = to_tensor(X_test_df[kernel_features])
    y_train = torch.tensor(y_train_scaled, dtype=torch.float32)
    y_val = torch.tensor(y_val_scaled, dtype=torch.float32)
    y_test = torch.tensor(y_test_scaled, dtype=torch.float32)
    train_task_ids = torch.tensor(train_task_ids, dtype=torch.long)
    val_task_ids = torch.tensor(val_task_ids, dtype=torch.long)
    test_task_ids = torch.tensor(test_task_ids, dtype=torch.long)
    num_tasks = int(Xdf["task_id"].nunique())

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "type",
                "lambda",
                "p",
                "sigma_t",
                "sigma_d",
                "MAE",
                "MSE",
                "RMSE",
                "MAPE",
                "R2",
                "time",
            ]
        )

    best_val_mae = float("inf")
    best_params = None
    best_model = None

    for i in range(n_trials):
        lambda_reg = lambda_p
        sigma_t = random.choice(sigma_t_grid)
        sigma_d = random.choice(sigma_d_grid)
        params = {
            "lambda_reg": lambda_reg,
            "p": p,
            "sigma_t": sigma_t,
            "sigma_d": sigma_d,
        }
        start_time = time.time()
        print(f"\n🔁 Trial {i+1}/{n_trials}: {params}")

        model = fit_multitask_model_lowrank(
            X_train.to(device),
            y_train.to(device),
            train_task_ids.to(device),
            num_tasks,
            params,
        )
        # Predict and inverse-transform for metrics
        print("✅ Model trained")
        print(f"A shape: {model.shared_basis.A.shape}, B shape: {model.B.shape}")
        if torch.cuda.is_available():
            print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6:.2f} MB")

        with torch.no_grad():
            y_val_pred_scaled = (
                predict_multitask(model, X_val.to(device), val_task_ids.to(device))
                .cpu()
                .numpy()
                .flatten()
            )
            y_val_true_scaled = y_val.cpu().numpy().flatten()
            val_task_ids_np = val_task_ids.cpu().numpy()

            y_val_pred = inverse_transform_per_task(
                y_val_pred_scaled, val_task_ids_np, scalers
            )
            y_val_true = inverse_transform_per_task(
                y_val_true_scaled, val_task_ids_np, scalers
            )

        elapsed = time.time() - start_time
        avg_metrics, _ = calculate_metrics_per_task(
            y_val_true, y_val_pred, val_task_ids_np, elapsed
        )
        print(
            f"📉 MAE: {avg_metrics['MAE']:.4f}, RMSE: {avg_metrics['RMSE']:.4f}, R²: {avg_metrics['R2']:.4f}"
        )

        # Log (append)
        with open(METRICS_PATH, "a", newline="") as flog:
            writer = csv.writer(flog)
            writer.writerow(
                [
                    "val",
                    lambda_reg,
                    p,
                    sigma_t,
                    sigma_d,
                    avg_metrics["MAE"],
                    avg_metrics["MSE"],
                    avg_metrics["RMSE"],
                    avg_metrics["MAPE"],
                    avg_metrics["R2"],
                    elapsed,
                ]
            )

        if avg_metrics["MAE"] < best_val_mae:
            best_val_mae = avg_metrics["MAE"]
            best_params = params.copy()
            best_model = model

    print(f"\n🏆 Best params: {best_params} with MAE: {best_val_mae:.4f}")
    torch.save(best_model, f"{MODEL_PATH}/best_model.pt")

    print("📦 Testing on test set...")
    with torch.no_grad():
        y_test_pred_scaled = (
            predict_multitask(best_model, X_test.to(device), test_task_ids.to(device))
            .cpu()
            .numpy()
            .flatten()
        )
        y_test_true_scaled = y_test.cpu().numpy().flatten()
        test_task_ids_np = test_task_ids.cpu().numpy()
        y_test_pred = inverse_transform_per_task(
            y_test_pred_scaled, test_task_ids_np, scalers
        )
        y_test_true = inverse_transform_per_task(
            y_test_true_scaled, test_task_ids_np, scalers
        )
        avg_test_metrics, _ = calculate_metrics_per_task(
            y_test_true, y_test_pred, test_task_ids_np, 0
        )
        save_predictions_per_task(
            y_test_true, y_test_pred, test_task_ids_np, "logs/prediction_data.csv"
        )

    with open(METRICS_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "test",
                best_params["lambda_reg"],
                best_params["p"],
                best_params["sigma_t"],
                best_params["sigma_d"],
                avg_test_metrics["MAE"],
                avg_test_metrics["MSE"],
                avg_test_metrics["RMSE"],
                avg_test_metrics["MAPE"],
                avg_test_metrics["R2"],
                0,
            ]
        )
    print("✅ Finished.")


if __name__ == "__main__":
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    random_search_training(
        "data/merged.csv",
        val_ratio=0.1,
        test_ratio=0.3,
        lambda_p=1,
        p=P,
        sigma_t_grid=[4.0],
        sigma_d_grid=[2.0],
        n_trials=1,
        device=DEVICE,
    )
