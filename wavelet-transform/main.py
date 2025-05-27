import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from model import SwtForecastingModel, swt_decompose, swt_reconstruct

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {DEVICE}")

def decompose_batch(batch):
    bands = [[] for _ in range(4)]
    for seq in batch:
        try:
            coeffs = swt_decompose(seq, level=2)
            a2, d2 = coeffs[0]
            d1 = coeffs[1][1]
            zero_pad = np.zeros_like(d1)
            for i, band in enumerate([a2, d2, d1, zero_pad]):
                bands[i].append(band)
        except Exception as e:
            print("❌ Error in SWT decomposition:", e)
            exit(1)
    return [torch.tensor(np.array(b), dtype=torch.float32).unsqueeze(-1).to(DEVICE) for b in bands]

if __name__ == "__main__":
    print("📥 Loading data...")
    df = pd.read_csv("mm79158.csv", parse_dates=['ts'])
    df.set_index('ts', inplace=True)
    data = df['vrednost'].values.reshape(-1, 1)

    print("🔄 Normalizing data with MinMaxScaler...")
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data).flatten()

    print("🧱 Creating sliding window sequences...")
    window = 12
    X, y = [], []
    for i in range(len(data_scaled) - window):
        X.append(data_scaled[i:i + window])
        y.append(data_scaled[i + window])
    X = np.array(X)
    y = np.array(y)

    print("✂️ Splitting into train/val/test sets...")
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, shuffle=False)
    print(f"Train samples: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    print("📐 Performing SWT decomposition...")
    X_train_tensors = decompose_batch(X_train)
    X_val_tensors = decompose_batch(X_val)
    X_test_tensors = decompose_batch(X_test)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

    train_dataset = TensorDataset(*X_train_tensors, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    print("🧠 Initializing model...")
    model = SwtForecastingModel(
        num_bands=4,
        input_size=1, time2vec_k=8, d_model=64, n_heads=4, d_ff=128, n_enc_layers=2, output_size=1
    ).to(DEVICE)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    print("🚀 Starting training...")
    train_losses = []
    for epoch in range(30):
        model.train()
        epoch_loss = 0
        for *x_bands, y_batch in train_loader:
            preds = model(x_bands).squeeze(-1)
            pred_mean = preds.mean(dim=1)
            loss = loss_fn(pred_mean, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * y_batch.size(0)
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f"[Epoch {epoch+1:02d}] Avg Train MSE Loss: {epoch_loss:.6f}")

    print("🔎 Evaluating on test set...")
    model.eval()
    with torch.no_grad():
        preds_test = model(X_test_tensors).squeeze(-1)  # [B, num_bands]
        pred_mean = preds_test.mean(dim=1)  # [B]
        pred_scaled = pred_mean.cpu().numpy()
        y_true_scaled = y_test_tensor.cpu().numpy()

        pred_rescaled = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        y_true_rescaled = scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_true_rescaled, pred_rescaled)
    mse = mean_squared_error(y_true_rescaled, pred_rescaled)
    mape = mean_absolute_percentage_error(y_true_rescaled, pred_rescaled)

    print("\n📊 Final Evaluation Results:")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"MAPE: {mape:.2%}")
