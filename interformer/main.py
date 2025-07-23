import os
import csv
import random
import torch
import torch.optim as optim
from tqdm import tqdm
import time
from model import InterFormer, pinball_loss
from data import (
    preprocess_all,
    prepare_prediction_window,
    prepare_interformer_dataloaders_and_prediction,
)
from util import evaluate_model


# === Config ===
INPUT_LEN = 24 * 14  # e.g., 1 day hourly
FORECAST_LEN = 1  # e.g., 12 hours
QUANTILES = [0.1, 0.5, 0.9]
TRIALS = 20
EPOCHS = 20
BATCH_SIZE = 64
LOGS_PATH = "logs/training_log.csv"
METRICS_PATH = "logs/training_eval.csv"

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
    "dropout": [0, 0.1, 0.3, 0.5],
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
        self.best_loss = float("inf")
        self.best_model = None

    def step(self, loss, model):
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_model = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def train_random_search(
    condition_df, quantiles, input_len, forecast_len, trials=5, epochs=30
):
    best_model = None
    best_val_loss = float("inf")

    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    for trial in range(trials):
        print(f"\n🔁 Trial {trial + 1}/{trials}")
        hp = sample_hyperparams()
        print(f"🧬 Sampled Hyperparameters: {hp}")

        # One-liner model name from hyperparams
        model_name = f"model_" + "_".join(f"{k}{v}" for k, v in hp.items())
        model_path = f"models/{model_name}.pt"

        # Data loaders
        train_loader, val_loader, _, _, scaler_y = (
            prepare_interformer_dataloaders_and_prediction(
                condition_df,
                input_len=input_len,
                forecast_len=forecast_len,
                batch_size=hp["batch_size"],
            )
        )

        # Input shapes for model creation
        x_cond_sample, x_pred_sample, _ = next(iter(train_loader))
        print("Train x_cond shape:", x_cond_sample.shape)
        print("Train x_pred shape:", x_pred_sample.shape)

        model = InterFormer(
            num_vars_cond=x_cond_sample.shape[2],
            num_vars_pred=x_pred_sample.shape[2],
            d_model=hp["d_model"],
            kernel_size=hp["kernel_size"],
            num_heads=hp["num_heads"],
            d_ff=hp["d_model"] * 4,
            num_layers=hp["num_layers"],
            horizon=forecast_len,
            quantiles=quantiles,
            dropout=hp["dropout"],
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=hp["learning_rate"])
        early_stopper = EarlyStopping(patience=5)

        for epoch in range(epochs):
            print(f"📚 Epoch {epoch + 1}/{epochs}")
            model.train()
            train_loss_sum = 0
            train_batches = 0
            start_epoch_time = time.time()

            for x_cond, x_pred, y in tqdm(train_loader, desc="Training"):
                x_cond, x_pred, y = x_cond.to(device), x_pred.to(device), y.to(device)
                if y.ndim == 1:
                    y = y.unsqueeze(1).expand(-1, forecast_len)
                elif y.ndim == 2 and y.size(1) == 1:
                    y = y.expand(-1, forecast_len)

                preds, *_ = model(x_cond, x_pred)
                loss = pinball_loss(y, preds, quantiles)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), hp["clip_value"])
                optimizer.step()

                train_loss_sum += loss.item()
                train_batches += 1

            avg_train_loss = train_loss_sum / train_batches if train_batches > 0 else 0
            end_epoch_time = time.time()
            # Validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for x_cond, x_pred, y in val_loader:
                    x_cond, x_pred, y = (
                        x_cond.to(device),
                        x_pred.to(device),
                        y.to(device),
                    )
                    if y.ndim == 1:
                        y = y.unsqueeze(1).expand(-1, forecast_len)
                    elif y.ndim == 2 and y.size(1) == 1:
                        y = y.expand(-1, forecast_len)

                    preds, *_ = model(x_cond, x_pred)
                    val_loss = pinball_loss(y, preds, quantiles)
                    val_losses.append(val_loss.item())

            avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0

            print(
                f"✅ Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}"
            )

            _, _, train_metrics = evaluate_model(
                model, train_loader, scaler_y, condition_df, "train"
            )
            _, _, val_metrics = evaluate_model(
                model, val_loader, scaler_y, condition_df, "val"
            )

            # Prepare CSV row
            train_row = [model_name, epoch + 1] + [
                train_metrics[k]
                for k in [
                    "type",
                    "inference",
                    "MAE",
                    "MSE",
                    "RMSE",
                    "MAPE",
                    "R2",
                    "MDA",
                    "Spearman",
                ]
            ]
            val_row = [model_name, epoch + 1] + [
                val_metrics[k]
                for k in [
                    "type",
                    "inference",
                    "MAE",
                    "MSE",
                    "RMSE",
                    "MAPE",
                    "R2",
                    "MDA",
                    "Spearman",
                ]
            ]

            with open(LOGS_PATH, "a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        trial + 1,
                        model_name,
                        epoch + 1,
                        time.strftime(
                            "%Y-%m-%d %H:%M:%S", time.localtime(start_epoch_time)
                        ),
                        time.strftime(
                            "%Y-%m-%d %H:%M:%S", time.localtime(end_epoch_time)
                        ),
                        avg_train_loss,
                        avg_val_loss,
                    ]
                )

            with open(METRICS_PATH, "a", newline="") as f:
                csv.writer(f).writerow(train_row)
                csv.writer(f).writerow(val_row)

        # Save model after full training
        torch.save(model, model_path)
        print(f"💾 Saved model: {model_path}")


# === Entry Point ===
if __name__ == "__main__":
    condition_df = preprocess_all(
        "data/mm79158.csv",
        "data/slovenia_hourly_weather.csv",
        "data/slovenian_holidays_2016_2018.csv",
    )

    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    with open(LOGS_PATH, "w", newline="") as f:
        csv.writer(f).writerow(
            [
                "trial",
                "model",
                "epoch",
                "epoch_start_time",
                "epoch_end_time",
                "train_loss",
                "val_loss",
            ]
        )

    with open(METRICS_PATH, "w", newline="") as f:
        csv.writer(f).writerow(
            [
                "model",
                "epoch",
                "type",
                "inference",
                "MAE",
                "MSE",
                "RMSE",
                "MAPE",
                "R2",
                "MDA",
                "Spearman",
            ]
        )

    try:
        best_model = train_random_search(
            condition_df,
            quantiles=QUANTILES,
            input_len=INPUT_LEN,
            forecast_len=FORECAST_LEN,
            trials=TRIALS,
            epochs=EPOCHS,
        )

    except Exception as e:
        print(f"🔥 Exception during training batch: {e}")
        import traceback

        traceback.print_exc()
