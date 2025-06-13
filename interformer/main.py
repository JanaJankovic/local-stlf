import os
import csv
import random
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from model import InterFormer, pinball_loss
from data import preprocess_all, prepare_prediction_window, prepare_interformer_dataloaders_and_prediction

# === Config ===
INPUT_LEN = 24            # e.g., 1 day hourly
FORECAST_LEN = 12         # e.g., 12 hours
QUANTILES = [0.1, 0.5, 0.9]
TRIALS = 5
EPOCHS = 30

# === Device Setup ===
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

# === Hyperparameter Space ===
HYPERPARAM_SPACE = {
    "learning_rate": [1e-4, 1e-3, 1e-2],
    "clip_value": [0.1, 1, 10],
    "batch_size": [64, 128, 256],
    "dropout": [0, 0.1, 0.15, 0.2],
    "d_model": [32, 64, 128],
    "num_layers": [1, 2, 4, 8],
    "num_heads": [1, 4, 8],
    "kernel_size": [2, 4, 6],
}

def sample_hyperparams():
    return {k: random.choice(v) for k, v in HYPERPARAM_SPACE.items()}

# === Early Stopping ===
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.best_model = None

    def step(self, loss, model):
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_model = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

# === Random Search Training ===
def train_random_search(condition_df, prediction_df, quantiles, input_len, forecast_len, trials=5, epochs=30):
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    best_model = None
    best_val_loss = float('inf')

    for trial in range(trials):
        print(f"\n🔁 Trial {trial+1}/{trials}")
        hp = sample_hyperparams()
        print(f"🧬 Sampled Hyperparameters: {hp}")

        # Paths for this trial
        trial_model_path = (
            f"models/model_trial{trial+1}_lr{hp['learning_rate']}_clip{hp['clip_value']}_bs{hp['batch_size']}_"
            f"dropout{hp['dropout']}_dmodel{hp['d_model']}_layers{hp['num_layers']}_"
            f"heads{hp['num_heads']}_kernel{hp['kernel_size']}.pth"
        )
        trial_logs_path = (
            f"logs/log_trial{trial+1}_lr{hp['learning_rate']}_clip{hp['clip_value']}_bs{hp['batch_size']}_"
            f"dropout{hp['dropout']}_dmodel{hp['d_model']}_layers{hp['num_layers']}_"
            f"heads{hp['num_heads']}_kernel{hp['kernel_size']}.csv"
        )
        with open(trial_logs_path, "w", newline="") as f:
            csv.writer(f).writerow(["trial", "epoch", "batch", "train_loss", "val_loss"])

        # Data preparation
        train_loader, val_loader, _, x_pred, _, _ = prepare_interformer_dataloaders_and_prediction(
            condition_df, prediction_df,
            input_len=input_len, forecast_len=forecast_len,
            batch_size=hp["batch_size"]
        )

        X_sample = next(iter(train_loader))[0]
        print("Train sample shape:", X_sample.shape)
        print("Prediction input shape:", x_pred.shape)

        model = InterFormer(
            num_vars_cond=X_sample.shape[2],
            num_vars_pred=x_pred.shape[2],
            d_model=hp["d_model"],
            kernel_size=hp["kernel_size"],
            num_heads=hp["num_heads"],
            d_ff=hp["d_model"] * 4,
            num_layers=hp["num_layers"],
            horizon=forecast_len,
            quantiles=quantiles,
            dropout=hp["dropout"]
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=hp["learning_rate"])
        early_stopper = EarlyStopping(patience=5)
        trial_best_loss = float('inf')

        for epoch in range(epochs):
            print(f"📚 Epoch {epoch+1}/{epochs}")
            model.train()
            train_loss_sum = 0
            train_batches = 0

            for batch_idx, (x_cond, y) in enumerate(tqdm(train_loader, desc="Training")):
                x_cond, y = x_cond.to(device), y.to(device)
                y = y.view(-1, forecast_len)  # Ensure shape is [B, H]
                x_pred_batch = x_pred.repeat(x_cond.size(0), 1, 1).to(device)

                preds, *_ = model(x_cond, x_pred_batch)
                loss = pinball_loss(y, preds, quantiles)

                optimizer.zero_grad()
                loss.backward()
                total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hp["clip_value"])
                optimizer.step()

                train_loss_sum += loss.item()
                train_batches += 1

            avg_train_loss = train_loss_sum / train_batches if train_batches > 0 else 0

            # Validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for x_cond, y in val_loader:
                    x_cond, y = x_cond.to(device), y.to(device)
                    y = y.view(-1, forecast_len)
                    x_pred_batch = x_pred.repeat(x_cond.size(0), 1, 1).to(device)
                    preds, *_ = model(x_cond, x_pred_batch)
                    val_loss = pinball_loss(y, preds, quantiles)
                    val_losses.append(val_loss.item())

            avg_val_loss = sum(val_losses) / len(val_losses) if len(val_losses) > 0 else 0

            print(f"✅ Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
            with open(trial_logs_path, "a", newline="") as f:
                csv.writer(f).writerow([trial+1, epoch+1, "", avg_train_loss, avg_val_loss])

            # Save best model per trial
            if avg_val_loss < trial_best_loss:
                trial_best_loss = avg_val_loss
                torch.save(model, trial_model_path)
                print(f"💾 Trial {trial+1}: Saved best model to {trial_model_path}")

            if early_stopper.step(avg_val_loss, model):
                print("⏹️ Early stopping triggered.")
                break

        # Track best model across all trials
        if trial_best_loss < best_val_loss:
            best_val_loss = trial_best_loss
            best_model = model

    print(f"\n🏁 Best Validation Loss across all trials: {best_val_loss:.4f}")
    return best_model


# === Entry Point ===
if __name__ == "__main__":
    condition_df = preprocess_all(
        "data/mm79158.csv",
        "data/slovenia_hourly_weather.csv",
        "data/slovenian_holidays_2016_2018.csv"
    )

    prediction_df = prepare_prediction_window(
        "data/slovenia_hourly_weather_future.csv",
        "data/slovenian_holidays_2019_2022.csv"
    )

    best_model = train_random_search(
        condition_df,
        prediction_df,
        quantiles=QUANTILES,
        input_len=INPUT_LEN,
        forecast_len=FORECAST_LEN,
        trials=TRIALS,
        epochs=EPOCHS
    )
