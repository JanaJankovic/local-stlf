import torch
import torch.nn as nn
from model import AutoformerForecast
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


def load_data(load_df, weather_df):
    load_df['hour'] = load_df['ts'].dt.floor('h')
    hourly_load = load_df.groupby('hour')['vrednost'].sum().reset_index()
    hourly_load.rename(columns={'hour': 'datetime', 'vrednost': 'load_kWh'}, inplace=True)

    weather_df = weather_df[['datetime', 'temperature_2m', 'relative_humidity_2m',
                             'windspeed_10m', 'winddirection_10m', 'precipitation']]

    merged_df = pd.merge(hourly_load, weather_df, on='datetime', how='inner')
    merged_df = merged_df.sort_values('datetime').reset_index(drop=True)
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

    return (
        np.array(load_seq),
        np.array(weather_hist_seq),
        np.array(weather_fore_seq),
        np.array(target_seq)
    )


def evaluate_model(model, X_load, X_weather_hist, X_weather_fore, y_true):
    model.eval()
    with torch.no_grad():
        preds = model(X_load, X_weather_hist, X_weather_fore)[:, 0, :].cpu().numpy()
        mae = mean_absolute_error(y_true, preds)
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        mape = mean_absolute_percentage_error(y_true, preds)
    return mae, rmse, mape


if __name__ == '__main__':
    df_load = pd.read_csv('mm79158.csv', parse_dates=['ts'])
    df_weather = pd.read_csv('slovenia_hourly_weather.csv', parse_dates=['datetime'])
    df = load_data(df_load, df_weather)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['load_kWh', 'temperature_2m', 'relative_humidity_2m',
                                           'windspeed_10m', 'winddirection_10m', 'precipitation']])

    split_1 = int(0.7 * len(scaled_data))
    split_2 = int(0.9 * len(scaled_data))
    train_data, val_data, test_data = scaled_data[:split_1], scaled_data[split_1:split_2], scaled_data[split_2:]

    seq_len = 24
    horizon = 24

    X_train_l, X_train_w_hist, X_train_w_fore, y_train = create_sequences(train_data, input_len=seq_len, forecast_horizon=horizon)
    X_val_l, X_val_w_hist, X_val_w_fore, y_val = create_sequences(val_data, input_len=seq_len, forecast_horizon=horizon)
    X_test_l, X_test_w_hist, X_test_w_fore, y_test = create_sequences(test_data, input_len=seq_len, forecast_horizon=horizon)

    train_x = torch.tensor(X_train_l).unsqueeze(1).float()
    train_w_hist = torch.tensor(X_train_w_hist).float()
    train_w_fore = torch.tensor(X_train_w_fore).float()
    train_y = torch.tensor(y_train).float()

    val_x = torch.tensor(X_val_l).unsqueeze(1).float()
    val_w_hist = torch.tensor(X_val_w_hist).float()
    val_w_fore = torch.tensor(X_val_w_fore).float()
    val_y = torch.tensor(y_val).float()

    C = 1
    num_factors = train_w_hist.shape[2]
    model = AutoformerForecast(in_channels=C, num_factors=num_factors, forecast_horizon=horizon)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    epochs = 20
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_x, train_w_hist, train_w_fore)[:, 0, :]  # [B, H]
        loss = criterion(output, train_y)
        loss.backward()
        optimizer.step()

        val_output = model(val_x, val_w_hist, val_w_fore)[:, 0, :]
        val_mae = mean_absolute_error(val_y.numpy(), val_output.detach().numpy())
        val_rmse = np.sqrt(mean_squared_error(val_y.numpy(), val_output.detach().numpy()))
        val_mape = mean_absolute_percentage_error(val_y.numpy(), val_output.detach().numpy())

        print(f'Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Val MAE: {val_mae:.4f}, Val RMSE: {val_rmse:.4f}')

    test_x = torch.tensor(X_test_l).unsqueeze(1).float()
    test_w_hist = torch.tensor(X_test_w_hist).float()
    test_w_fore = torch.tensor(X_test_w_fore).float()
    test_y_tensor = torch.tensor(y_test).float()

    test_output = model(test_x, test_w_hist, test_w_fore)[:, 0, :]
    test_mae = mean_absolute_error(test_y_tensor.numpy(), test_output.detach().numpy())
    test_rmse = np.sqrt(mean_squared_error(test_y_tensor.numpy(), test_output.detach().numpy()))
    test_mape = mean_absolute_percentage_error(test_y_tensor.numpy(), test_output.detach().numpy())

    print(f'Test MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, MAPE: {test_mape:.4f}')
