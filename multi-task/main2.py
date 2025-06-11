import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from model import MultiTaskOKL  # replace with actual import

# === DATA LOADING & PREPROCESSING ===
class SmartMeterDataset(Dataset):
    def __init__(self, df):
        self.X = torch.tensor(df[["time", "day_of_year", "day_type"]].values, dtype=torch.float32)
        self.y = torch.tensor(df["value"].values, dtype=torch.float32)
        self.task_ids = torch.tensor(df["task_id"].values, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.task_ids[idx]

def load_and_preprocess(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    
    # Downsample to 3-hour intervals (8 measurements per day)
    df["time_slot"] = (df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute) // 180  # 3-hour slots (0 to 7)
    df = df.groupby([df["timestamp"].dt.date, "task_id", "time_slot"]).mean().reset_index()
    df["time"] = df["time_slot"] * 3  # Convert to hours (0, 3, 6, ..., 21)
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    df["day_type"] = df["timestamp"].dt.weekday
    df["task_id"], _ = pd.factorize(df["meter_id"])
    
    # Drop temporary columns and handle DST if needed (simplified here)
    df = df.drop(columns=["time_slot", "timestamp"])
    return df

# === TRAINING FUNCTION WITH ALTERNATING MINIMIZATION ===
def train_model(model, loader, optimizer_a, optimizer_b, loss_fn, lambda_reg, device, max_iter=10):
    model.train()
    for X, y, task_ids in loader:
        X, y = X.to(device), y.to(device)
        batch_size = X.shape[0]

        # Alternating minimization loop
        for _ in range(max_iter):
            # Update a (coefficients in SharedBasisFunctions)
            optimizer_a.zero_grad()
            preds = model(X)
            y_hat = preds[torch.arange(batch_size), task_ids]
            L = model.b @ model.b.T  # Implicit output kernel
            reg_term = lambda_reg * torch.trace(L)
            loss = loss_fn(y_hat, y) + reg_term
            loss.backward()
            optimizer_a.step()

            # Update b (coefficients in MultiTaskOKL)
            optimizer_b.zero_grad()
            preds = model(X)
            y_hat = preds[torch.arange(batch_size), task_ids]
            L = model.b @ model.b.T
            reg_term = lambda_reg * torch.trace(L)
            loss = loss_fn(y_hat, y) + reg_term
            loss.backward()
            optimizer_b.step()

# === EVALUATION FUNCTION: MAPE and MNAE (Eq. 19–20) ===
def evaluate_model(model, loader, device):
    model.eval()
    all_y, all_y_hat = [], []
    with torch.no_grad():
        for X, y, task_ids in loader:
            X = X.to(device)
            preds = model(X)
            y_hat = preds[torch.arange(len(task_ids)), task_ids].cpu().numpy()
            all_y.extend(y.numpy())
            all_y_hat.extend(y_hat)

    y = np.array(all_y)
    y_hat = np.array(all_y_hat)
    mape = np.mean(np.abs((y - y_hat) / np.maximum(y, 1e-3))) * 100  # Eq. 19
    mnae = np.mean(np.abs(y - y_hat)) / np.mean(np.abs(y))           # Eq. 20
    return mape, mnae

# === MAIN ENTRYPOINT ===
def run_training_pipeline(csv_path):
    df = load_and_preprocess(csv_path)
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

    train_data = SmartMeterDataset(train_df)
    test_data = SmartMeterDataset(test_df)

    train_loader = DataLoader(train_data, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1024)

    num_tasks = df["task_id"].nunique()
    X_train = torch.tensor(train_df[["time", "day_of_year", "day_type"]].values, dtype=torch.float32)
    model = MultiTaskOKL(X_train=X_train, num_tasks=num_tasks, p=10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Separate optimizers for a and b parameters
    optimizer_a = torch.optim.Adam([param for param in model.shared_basis.parameters()], lr=0.01)
    optimizer_b = torch.optim.Adam([param for param in model.parameters() if param not in model.shared_basis.parameters()], lr=0.01)
    loss_fn = nn.MSELoss()
    lambda_reg = 0.01  # Regularization parameter (tunable)

    print("Training model with alternating minimization...")
    for epoch in range(10):  # Increase as needed
        train_model(model, train_loader, optimizer_a, optimizer_b, loss_fn, lambda_reg, device)
        mape, mnae = evaluate_model(model, test_loader, device)
        print(f"Epoch {epoch+1}: MAPE = {mape:.2f}%, MNAE = {mnae:.4f}")

    print("\nFinal Evaluation on Test Set:")
    final_mape, final_mnae = evaluate_model(model, test_loader, device)
    print(f"Final MAPE: {final_mape:.2f}%")
    print(f"Final MNAE: {final_mnae:.4f}")

if __name__ == "__main__":
    run_training_pipeline("mm79158.csv")