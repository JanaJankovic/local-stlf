import torch
import torch.nn as nn
from model import AutoformerForecast
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from torch.utils.data import TensorDataset, DataLoader

def load_data(load_df, weather_df):
    load_df['hour'] = load_df['ts'].dt.floor('h')
    hourly_load = load_df.groupby('hour')['vrednost'].sum().reset_index()
    hourly_load.rename(columns={'hour': 'datetime', 'vrednost': 'load_kWh'}, inplace=True)

    weather_df = weather_df[['datetime', 'temperature_2m', 'relative_humidity_2m',
                             'windspeed_10m', 'winddirection_10m', 'precipitation']]

    merged_df = pd.merge(hourly_load, weather_df, on='datetime', how='inner')
    merged_df = merged_df.sort_values('datetime').reset_index(drop=True)
    print("Loaded and merged data.")
    return merged_df

def create_sequences(data, input_len=24, forecast_horizon=24):
    load_seq = []
    weather_hist_seq = []
    weather_fore_seq = []
    target_seq = []

    total_len = input_len + forecast_horizon

    for i in range(len(data) - total_len):
        input_block = data[i : i + input_len]
        forecast_block = data[i + input_len : i + total_len]

        load_seq.append(input_block[:, 0])
        weather_hist_seq.append(input_block[:, 1:])
        weather_fore_seq.append(forecast_block[:, 1:])
        target_seq.append(forecast_block[:, 0])

    print(f"Created {len(load_seq)} sequences of length {input_len} with horizon {forecast_horizon}.")
    return (
        np.array(load_seq),
        np.array(weather_hist_seq),
        np.array(weather_fore_seq),
        np.array(target_seq)
    )

def evaluate_model(model, dataloader, y_true, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, wh, wf in dataloader:
            xb, wh, wf = xb.to(device), wh.to(device), wf.to(device)
            out = model(xb, wh, wf)[:, 0, :].cpu().numpy()
            preds.append(out)
    preds = np.concatenate(preds, axis=0)
    mae = mean_absolute_error(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mape = mean_absolute_percentage_error(y_true, preds)
    print("Evaluation results:")
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}")
    return mae, rmse, mape

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    df_load = pd.read_csv('mm79158.csv', parse_dates=['ts'])
    df_weather = pd.read_csv('slovenia_hourly_weather.csv', parse_dates=['datetime'])
    df = load_data(df_load, df_weather)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['load_kWh', 'temperature_2m', 'relative_humidity_2m',
                                           'windspeed_10m', 'winddirection_10m', 'precipitation']])
    print("Scaled data.")

    split_1 = int(0.7 * len(scaled_data))
    split_2 = int(0.9 * len(scaled_data))
    train_data, val_data, test_data = scaled_data[:split_1], scaled_data[split_1:split_2], scaled_data[split_2:]
    print(f"Data split into train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)}")

    seq_len = 24
    horizon = 24

    X_train_l, X_train_w_hist, X_train_w_fore, y_train = create_sequences(train_data, seq_len, horizon)
    X_val_l, X_val_w_hist, X_val_w_fore, y_val = create_sequences(val_data, seq_len, horizon)
    X_test_l, X_test_w_hist, X_test_w_fore, y_test = create_sequences(test_data, seq_len, horizon)

    # Convert to tensors
    train_ds = TensorDataset(torch.tensor(X_train_l).unsqueeze(1).float(),
                             torch.tensor(X_train_w_hist).float(),
                             torch.tensor(X_train_w_fore).float(),
                             torch.tensor(y_train).float())

    val_ds = TensorDataset(torch.tensor(X_val_l).unsqueeze(1).float(),
                           torch.tensor(X_val_w_hist).float(),
                           torch.tensor(X_val_w_fore).float(),
                           torch.tensor(y_val).float())

    test_ds = TensorDataset(torch.tensor(X_test_l).unsqueeze(1).float(),
                            torch.tensor(X_test_w_hist).float(),
                            torch.tensor(X_test_w_fore).float())

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=256)
    test_loader = DataLoader(test_ds, batch_size=256)

    C = 1
    num_factors = X_train_w_hist.shape[2]
    model = AutoformerForecast(in_channels=C, num_factors=num_factors, forecast_horizon=horizon).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print("Starting training loop...")
    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, wh, wf, yb in train_loader:
            xb, wh, wf, yb = xb.to(device), wh.to(device), wf.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb, wh, wf)[:, 0, :]
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Train Loss = {total_loss / len(train_loader):.4f}")

        val_preds = []
        val_y_all = []
        model.eval()
        with torch.no_grad():
            for xb, wh, wf, yb in val_loader:
                xb, wh, wf = xb.to(device), wh.to(device), wf.to(device)
                out = model(xb, wh, wf)[:, 0, :]
                val_preds.append(out.cpu())
                val_y_all.append(yb)
        val_preds = torch.cat(val_preds).numpy()
        val_y_all = torch.cat(val_y_all).numpy()
        val_mae = mean_absolute_error(val_y_all, val_preds)
        val_rmse = np.sqrt(mean_squared_error(val_y_all, val_preds))
        val_mape = mean_absolute_percentage_error(val_y_all, val_preds)
        print(f"Epoch {epoch+1}, Val MAE: {val_mae:.4f}, Val RMSE: {val_rmse:.4f}, Val MAPE: {val_mape:.4f}")

    print("Training complete. Evaluating on test set...")
    test_y_tensor = torch.tensor(y_test).float()
    evaluate_model(model, test_loader, test_y_tensor.numpy(), device)