import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from power_net import PowerNet
from itertools import product

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_calendar_features(df, timestamp_col='ts'):
    print("🗓️ Creating calendar features...")
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df['day_of_month'] = df[timestamp_col].dt.day
    df['day_of_week'] = df[timestamp_col].dt.weekday
    df['hour_of_day'] = df[timestamp_col].dt.hour
    df['period_of_day'] = df['hour_of_day'].apply(lambda h: 1 if 6 <= h < 18 else 0)
    df['is_weekend'] = df['day_of_week'].apply(lambda d: 1 if d >= 5 else 0)
    print("✅ Calendar features added.")
    return df


def preprocess_weather_data(path, missing_threshold=0.15):
    print(f"🌦️ Preprocessing weather data from {path}...")
    df = pd.read_csv(path, parse_dates=['time'])
    df.set_index('time', inplace=True)
    missing_ratio = df.isna().mean()
    cols_to_drop = missing_ratio[missing_ratio > missing_threshold].index.tolist()
    if cols_to_drop:
        print(f"⚠️ Dropping columns due to excessive missing data: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)
    df = df.interpolate(method='time', limit_direction='both')
    print("✅ Weather data processed.")
    return df


def join_calendar_and_weather(load_df, weather_df, timestamp_col='ts'):
    print("🔗 Joining calendar features with weather data...")
    load_df[timestamp_col] = pd.to_datetime(load_df[timestamp_col])
    load_df = load_df.set_index(timestamp_col)
    merged_df = load_df.join(weather_df, how='inner')
    print("✅ Data joined successfully.")
    return merged_df.reset_index().rename(columns={'index': 'ts'})


def prepare_powernet_data(df, target_col='vrednost', timestamp_col='ts', test_size=0.2, val_size=0.1, seq_len=48, batch_size=64):
    print("📦 Preparing data for PowerNet...")
    df = df.drop(columns=[timestamp_col])
    y = df[[target_col]].copy()
    X = df.drop(columns=[target_col])
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_ratio, shuffle=False)

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    y_train_scaled = target_scaler.fit_transform(y_train)
    X_val_scaled = feature_scaler.transform(X_val)
    y_val_scaled = target_scaler.transform(y_val)
    X_test_scaled = feature_scaler.transform(X_test)
    y_test_scaled = target_scaler.transform(y_test)

    def to_tensor_split(X, y):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y.values if isinstance(y, pd.DataFrame) else y, dtype=torch.float32)
        seq_x = X[:, :seq_len].unsqueeze(-1)
        meta_x = X[:, seq_len:]
        return seq_x, meta_x, y

    seq_train, meta_train, y_train = to_tensor_split(X_train_scaled, y_train_scaled)
    seq_val, meta_val, y_val = to_tensor_split(X_val_scaled, y_val_scaled)
    seq_test, meta_test, y_test = to_tensor_split(X_test_scaled, y_test_scaled)

    train_loader = DataLoader(TensorDataset(seq_train, meta_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(seq_val, meta_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(seq_test, meta_test, y_test), batch_size=batch_size)

    print("✅ Data ready for training.")
    return train_loader, val_loader, test_loader, feature_scaler, target_scaler


def train_powernet(model, train_loader, val_loader, epochs=30, lr=1e-3, device='cuda'):
    print("🚀 Training PowerNet...")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for seq_x, meta_x, y in train_loader:
            seq_x, meta_x, y = seq_x.to(device), meta_x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(seq_x, meta_x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seq_x, meta_x, y in val_loader:
                seq_x, meta_x, y = seq_x.to(device), meta_x.to(device), y.to(device)
                preds = model(seq_x, meta_x)
                val_loss += criterion(preds, y).item()

        print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")


def search_powernet_hyperparams(train_loader, val_loader, input_size_lstm, input_size_meta):
    print("🔍 Starting hyperparameter search...")
    lstm_hidden_options = [64, 128, 256, 512]
    mlp_hidden1_options = [32, 64]
    mlp_hidden2_options = [16, 32]
    final_hidden_options = [32, 64]
    dropout_options = [0.2, 0.3]
    lr_options = [1e-3, 5e-4]
    epochs = 30

    best_model = None
    best_val_loss = float('inf')
    best_params = {}

    for lstm_h, mlp_h1, mlp_h2, final_h, dropout, lr in product(
        lstm_hidden_options,
        mlp_hidden1_options,
        mlp_hidden2_options,
        final_hidden_options,
        dropout_options,
        lr_options
    ):
        print(f"\n🔧 Trying config: LSTM={lstm_h}, MLP1={mlp_h1}, MLP2={mlp_h2}, FINAL={final_h}, dropout={dropout}, lr={lr}")

        model = PowerNet(
            input_size_meta=input_size_meta,
            input_size_lstm=input_size_lstm,
            lstm_hidden=lstm_h,
            lstm_layers=2,
            mlp_hidden1=mlp_h1,
            mlp_hidden2=mlp_h2,
            final_hidden=final_h,
            dropout=dropout
        )

        train_powernet(model, train_loader, val_loader, epochs=epochs, lr=lr, device=device)

        model.eval()
        total_val_loss = 0
        criterion = nn.MSELoss()
        with torch.no_grad():
            for seq_x, meta_x, y in val_loader:
                seq_x, meta_x, y = seq_x.to(device), meta_x.to(device), y.to(device)
                preds = model(seq_x, meta_x)
                total_val_loss += criterion(preds, y).item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"📉 Validation Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model
            best_params = {
                'lstm_hidden': lstm_h,
                'mlp_hidden1': mlp_h1,
                'mlp_hidden2': mlp_h2,
                'final_hidden': final_h,
                'dropout': dropout,
                'lr': lr
            }

    print("\n🏆 Best config found:")
    print(best_params)
    return best_model, best_params


def evaluate_powernet(model, test_data, scaler, device='cpu'):
    print("\n🧪 Evaluating model on test set...")
    model.eval()
    X_seq_test, X_per_test, y_test = [torch.tensor(x, dtype=torch.float32).to(device) for x in test_data]

    with torch.no_grad():
        preds = model(X_seq_test, X_per_test).cpu().numpy()

    y_true = y_test.cpu().numpy()
    y_pred = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    y_true_inv = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_true_inv, y_pred)
    mse = mean_squared_error(y_true_inv, y_pred)
    mape = mean_absolute_percentage_error(y_true_inv, y_pred)

    print("\n📊 Evaluation Metrics:")
    print(f"✅ MAE:  {mae:.4f}")
    print(f"✅ MSE:  {mse:.4f}")
    print(f"✅ MAPE: {mape:.4f}")


if __name__ == "__main__":
    print("🚚 Loading data...")
    data = create_calendar_features(pd.read_csv('mm79158.csv'), 'ts')
    wdf = preprocess_weather_data('slovenia_weather_averaged.csv')
    df = join_calendar_and_weather(data, wdf, 'ts')

    print("📊 Columns in final DataFrame:")
    print(df.columns)

    # After prepare_powernet_data(...)
    train_loader, val_loader, test_loader, feature_scaler, target_scaler = prepare_powernet_data(df)

    # Extract input sizes from a batch
    sample_seq, sample_meta, _ = next(iter(train_loader))
    input_size_lstm = 1   # should be 48
    input_size_meta = sample_meta.shape[1]  # correct number of meta features

    best_model, best_config = search_powernet_hyperparams(
        train_loader, val_loader, input_size_lstm, input_size_meta
    )

    evaluate_powernet(best_model, test_loader.dataset.tensors, target_scaler)
