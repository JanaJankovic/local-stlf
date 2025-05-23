from interformer import InterFormer, pinball_loss
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
from tqdm import tqdm


import gc
gc.collect()
torch.cuda.empty_cache()  # if using GPU

print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HYPERPARAM_SPACE = {
    "learning_rate": [1e-4, 1e-3, 1e-2],
    "clip_value": [0.1, 1, 10],
    "batch_size": [64, 128, 256], # different from the paper, since memory is limited
    "dropout": [0, 0.1, 0.3, 0.5],
    "d_model": [32, 64, 128],
    "num_layers": [1, 2, 4, 8],
    "num_heads": [1, 4, 8],
    "kernel_size": [2, 4, 6]
}


def data_preprocessing(path):
    print("🔄 Parsing timestamps and interpolating...")
    df = pd.read_csv(path, sep=';', decimal=',')
    df['ts'] = pd.to_datetime(df['ts'])
    df.set_index('ts', inplace=True)

    # Resample to 30-minute intervals and interpolate missing values
    df_resampled = df.resample('30min').mean()
    df_resampled['vrednost'] = df_resampled['vrednost'].interpolate(method='time')

    # === Extract time-based features ===
    df_resampled['hour_48'] = df_resampled.index.map(lambda x: (x.hour * 2 + x.minute // 30) + 1)
    df_resampled['day_of_week'] = df_resampled.index.dayofweek
    df_resampled['is_weekend'] = df_resampled['day_of_week'].isin([5, 6]).astype(int)

    # === Load holidays and tag rows ===
    holidays = pd.read_csv("slovenian_holidays_2016_2018.csv")
    holidays['holiday_date'] = pd.to_datetime(holidays['holiday_date'])
    holiday_set = set(holidays['holiday_date'].dt.normalize())
    df_resampled['is_holiday'] = df_resampled.index.normalize().isin(holiday_set).astype(int)

    # === One-hot encode categorical known covariates ===
    known_covariates = ['hour_48', 'day_of_week', 'is_holiday', 'is_weekend']
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(df_resampled[known_covariates])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(known_covariates),
        index=df_resampled.index
    )

    # === Combine final DataFrame ===
    df_final = pd.concat([df_resampled[['vrednost']], encoded_df], axis=1)
    df_final = df_final.sort_index()
    print(f"✅ Final DataFrame shape: {df_final.shape}")

    return df_final

def create_dataset(df):
    print("📐 Creating sliding windows...")
    t0, tau = 336, 48  # 7 days input, 1 day prediction

    X, y = [], []
    for i in range(t0, len(df) - tau):
        X.append(df.iloc[i - t0:i].values)  # [t0, n_features]
        y.append(df.iloc[i:i + tau]['vrednost'].values)  # [tau]
    X = np.array(X)
    y = np.array(y)
    print(f"✔️ X shape: {X.shape}, y shape: {y.shape}")

    print("📊 Splitting into train/val/test...")
    # === Train/test split ===
    X_trainval_raw, X_test_raw, y_trainval_raw, y_test_raw = train_test_split(
        X, y, test_size=0.2, shuffle=False  # time series: don't shuffle
    )

    # === Validation split from training data ===
    X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
        X_trainval_raw, y_trainval_raw, test_size=0.1, shuffle=False
    )
    print(f"→ Train: {X_train_raw.shape}, Val: {X_val_raw.shape}, Test: {X_test_raw.shape}")

    # === Scale input features ===
    print("🔃 Scaling input features...")
    scaler = MinMaxScaler()
    n_features = X.shape[2]

    X_train_scaled = scaler.fit_transform(X_train_raw.reshape(-1, n_features)).reshape(X_train_raw.shape)
    X_val_scaled = scaler.transform(X_val_raw.reshape(-1, n_features)).reshape(X_val_raw.shape)
    X_test_scaled = scaler.transform(X_test_raw.reshape(-1, n_features)).reshape(X_test_raw.shape)

    # === Convert to tensors ===
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val   = torch.tensor(X_val_scaled, dtype=torch.float32)
    X_test  = torch.tensor(X_test_scaled, dtype=torch.float32)

    y_train = torch.tensor(y_train_raw, dtype=torch.float32)
    y_val   = torch.tensor(y_val_raw, dtype=torch.float32)
    y_test  = torch.tensor(y_test_raw, dtype=torch.float32)

    print(f"✅ Scaled: {X_train.shape[2]} features")
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

def sample_hyperparams():
    return {k: random.choice(v) for k, v in HYPERPARAM_SPACE.items()}

# === Early stopping helper ===
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

# === Random search training ===
def train_random_search(X_train, y_train, X_val, y_val, quantiles, tau, trials=5):
    print(f"\n🔎 Starting random search with {trials} trials...")

    best_model = None
    best_val_loss = float('inf')

    for trial in range(trials):
        print(f"\n🔍 Trial {trial+1}/{trials}")

        # Sample hyperparameters
        hp = sample_hyperparams()
        print(f"🧬 Sampled Hyperparameters: {hp}")

        # Prepare DataLoaders
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=hp["batch_size"], shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=hp["batch_size"])

        print(f"🛠️  Building InterFormer model...")
        # Create model
        model = InterFormer(
            num_vars=X_train.shape[2],
            d_model=hp["d_model"],
            kernel_size=hp["kernel_size"],
            num_heads=hp["num_heads"],
            d_ff=hp["d_model"] * 4,  # paper says d_ff = 4 * d_model
            num_layers=hp["num_layers"],
            horizon=tau,
            quantiles=quantiles
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=hp["learning_rate"])
        early_stopper = EarlyStopping(patience=5)

        for epoch in range(30):
            print(f"\n📚 Epoch {epoch+1}/30")
            model.train()
            total_loss = 0
            loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{30}")
            for batch_idx, (xb, yb) in loop:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = model(xb)
                loss = pinball_loss(yb, preds, quantiles)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), hp["clip_value"])  # Gradient clipping
                optimizer.step()
                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            # Evaluate on validation set
            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb)
                    val_loss = pinball_loss(yb, preds, quantiles)
                    val_losses.append(val_loss.item())

            val_loss = sum(val_losses) / len(val_losses)
            print(f"🔍 Validation Loss = {val_loss:.4f}")
            
            print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

            if early_stopper.step(val_loss, model):
                print("🛑 Early stopping triggered.")
                break

        # Save best model from trial
        if val_loss < best_val_loss:
            print("🏆 New best model found.")
            best_val_loss = val_loss
            best_model = model
            best_model.load_state_dict(early_stopper.best_model)

    print(f"\n✅ Best Validation Loss: {best_val_loss:.4f}")
    return best_model


def model_evaluation(model, X_test, y_test, scaler):
    print("\n📈 Evaluating model on test set...")
    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(device))  # [batch, quantiles, horizon]
        preds_median = preds[:, 1, :]  # select 0.5 quantile

    # Inverse scale
    preds_inv = scaler.inverse_transform(preds_median.numpy())
    y_true_inv = scaler.inverse_transform(y_test.numpy())

    # === Evaluation ===
    mae = mean_absolute_error(y_true_inv, preds_inv)
    mse = mean_squared_error(y_true_inv, preds_inv)
    mape = mean_absolute_percentage_error(y_true_inv, preds_inv)
    
    print("📊 Final Evaluation Metrics:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"MAPE: {mape:.4f}")


if __name__ == "__main__":
    path = "mm79158.csv"
    print("Loading data...")
    df = data_preprocessing(path)
    print("Data loaded. Preprocessing...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = create_dataset(df)
    print("Data preprocessed. Starting training...")
    model = train_random_search(X_train, y_train, X_val, y_val, quantiles=[0.1, 0.5, 0.9], tau=48)
    print("Training complete. Evaluating model...")
    model_evaluation(model, X_test, y_test, scaler)
