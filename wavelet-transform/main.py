import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from model import SwtForecastingModel, swt_decompose, EarlyStopping
import argparse
import matplotlib.pyplot as plt
import os
import pywt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {DEVICE}")

def swt_reconstruct_torch(coeffs, wavelet='db2'):
    """
    coeffs: list of (approximation, detail) tuples, each of shape [s]
    This function does not preserve gradients (due to pywt), use only in evaluation!
    """
    coeffs_np = [(a.detach().cpu().numpy(), d.detach().cpu().numpy()) for (a, d) in coeffs]
    recon_np = pywt.iswt(coeffs_np, wavelet)
    return torch.tensor(recon_np, dtype=torch.float32).to(DEVICE)


def decompose_batch(batch, level=2):
    required_multiple = 2 ** level
    bands = [[] for _ in range(4)]
    for seq in batch:
        try:
            seq = np.asarray(seq).flatten()
            valid_len = (len(seq) // required_multiple) * required_multiple
            if valid_len % 2 != 0:
                valid_len -= 1  # ensure even length
            if valid_len < required_multiple:
                raise ValueError(f"Sequence too short for level={level}: got length={len(seq)}, need at least {required_multiple}")
            seq = seq[:valid_len]
            coeffs = swt_decompose(seq, level=level)
            a2, d2 = coeffs[0]
            d1 = coeffs[1][1]
            zero_pad = np.zeros_like(d1)
            for i, band in enumerate([a2, d2, d1, zero_pad]):
                bands[i].append(band)
        except Exception as e:
            print("❌ Error in SWT decomposition:", e)
            print(f"Sequence length: {len(seq)} | Level: {level} | Required multiple: {required_multiple}")
            exit(1)
    return [torch.tensor(np.array(b), dtype=torch.float32).unsqueeze(-1).to(DEVICE) for b in bands]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SWT Forecasting Model")
    parser.add_argument("-t", type=float, default=0.2, help="Proportion of data used for testing and validation (split equally)")
    parser.add_argument("-s", type=int, default=24, help="Steps for iterative forecasting")
    parser.add_argument("-w", type=int, default=48, help="Sliding window size")
    parser.add_argument("--level", type=int, default=3, help="SWT decomposition level")
    args = parser.parse_args()

    print("📅 Loading data...")
    df = pd.read_csv("mm79158.csv", parse_dates=['ts'])
    df.set_index('ts', inplace=True)
    data = df['vrednost'].values.reshape(-1, 1)

    print("📊 Creating sliding window sequences...")
    window = args.w
    X_raw, y_raw = [], []
    for i in range(len(data) - window - args.s + 1):
        X_raw.append(data[i:i + window])
        y_raw.append(data[i + window:i + window + args.s])

    X_raw = np.array(X_raw)
    y_raw = np.array(y_raw)

    print("✂️ Splitting into train/val/test sets...")
    total_size = len(X_raw)
    test_size = int(total_size * args.t)
    val_size = test_size
    train_size = total_size - test_size - val_size

    X_train, y_train = X_raw[:train_size], y_raw[:train_size]
    X_val, y_val = X_raw[train_size:train_size + val_size], y_raw[train_size:train_size + val_size]
    X_test, y_test = X_raw[train_size + val_size:], y_raw[train_size + val_size:]

    print("🔄 Normalizing with MinMaxScaler (fit on training data)...")
    scaler = MinMaxScaler()
    X_train_reshaped = X_train.reshape(-1, 1)
    scaler.fit(X_train_reshaped)
    X_train = scaler.transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
    y_train = scaler.transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
    y_val = scaler.transform(y_val.reshape(-1, 1)).reshape(y_val.shape)
    y_test = scaler.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

    print("🔄 Performing SWT decomposition...")
    X_train_tensors = decompose_batch(X_train, level=args.level)
    X_val_tensors = decompose_batch(X_val, level=args.level)
    X_test_tensors = decompose_batch(X_test, level=args.level)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

    train_dataset = TensorDataset(*X_train_tensors, y_train_tensor)
    val_dataset = TensorDataset(*X_val_tensors, y_val_tensor)

    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)

    print("🧠 Initializing model...")
    model = SwtForecastingModel(
        num_bands=4,
        input_size=1, time2vec_k=8, d_model=64, n_heads=4, d_ff=128, n_enc_layers=2, output_size=1, forecast_steps=args.s
    ).to(DEVICE)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    early_stopper = EarlyStopping(patience=10, min_delta=1e-4)

    print("🚀 Starting training...")
    for epoch in range(100):
        model.train()
        train_loss = 0
        for *x_bands, y_batch in train_loader:
            preds = model(x_bands)  # [B, 4, s, 1]
            preds = preds.squeeze(-1)  # [B, 4, s]

            recon_batch = []
            for b in range(preds.shape[0]):
                a2 = preds[b, 0, :]  # shape: [s]
                d2 = preds[b, 1, :]
                d1 = preds[b, 2, :]
                recon = a2 + d2 + d1  # simple additive approximation
                recon_batch.append(recon)

            recon_batch = torch.stack(recon_batch).unsqueeze(-1)  # [B, s, 1]
            loss = loss_fn(recon_batch, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * y_batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for *x_bands, y_batch in val_loader:
                preds = model(x_bands).squeeze(-1)
                recon_batch = []
                for b in range(preds.shape[0]):
                    coeffs = [
                        (preds[b, 0], preds[b, 1]),
                        (torch.zeros_like(preds[b, 1]), preds[b, 2])
                    ]
                    recon = swt_reconstruct_torch(coeffs)
                    recon_batch.append(recon[-args.s:])
                recon_batch = torch.stack(recon_batch).to(DEVICE)
                loss = loss_fn(recon_batch, y_batch.squeeze(-1))
                val_loss += loss.item() * y_batch.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"[Epoch {epoch+1:02d}] Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print("⏹️ Early stopping triggered.")
            model.load_state_dict(early_stopper.best_state_dict)
            break

    print("🔎 Evaluating on test set...")
    model.eval()
    preds = []

    with torch.no_grad():
        for i in range(len(X_test)):
            input_seq = X_test[i]  # [window]
            input_bands = decompose_batch([input_seq], level=args.level)
            band_outputs = model(input_bands)  # [1, num_bands, s, 1]
            band_outputs = band_outputs.squeeze(0).squeeze(-1)  # [num_bands, s]

            # Reconstruct signal using inverse SWT
            coeffs = [
                (band_outputs[0], band_outputs[1]),
                (torch.zeros_like(band_outputs[1]), band_outputs[2])
            ]
            recon = swt_reconstruct_torch(coeffs)  # [signal_len]

            preds.append(recon[-1])  # use last step only (or first if you prefer)

    preds = torch.stack(preds).cpu().numpy().reshape(-1, 1)  # shape [T, 1]
    y_true = y_test[:len(preds)].reshape(-1, 1)  # match length exactly

    # Inverse scale
    pred_rescaled = scaler.inverse_transform(preds).flatten()
    y_true_rescaled = scaler.inverse_transform(y_true).flatten()



    mae = mean_absolute_error(y_true_rescaled, pred_rescaled)
    mse = mean_squared_error(y_true_rescaled, pred_rescaled)
    mape = mean_absolute_percentage_error(y_true_rescaled, pred_rescaled)

    print("\n📊 Final Evaluation Results:")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"MAPE: {mape:.2%}")

    print("📈 Saving forecast plot (y_true vs predicted)...")

    plt.figure(figsize=(30, 7))
    plt.plot(y_true_rescaled, label="True", color="blue", linewidth=0.5)
    plt.plot(pred_rescaled, label="Predicted", color="red", linewidth=0.5)
    plt.title("Predicted vs True Values (Test Horizon)")
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/forecast_y_only_thin.png", dpi=300)
    plt.close()