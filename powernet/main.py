import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from itertools import product
from model import PowerNet
from data import create_calendar_features, preprocess_weather_data, join_calendar_and_weather, build_sequence_data, prepare_powernet_data
from util import log_loss_values, log_training_metrics
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def train_powernet(model, model_name, scaler, train_loader, val_loader, epochs=100, lr=1e-3, device='cuda', patience=10):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    early_stopper = EarlyStopping(patience=patience)

    for epoch in range(1, epochs + 1):
        start_epoch_time = time.time()

        # === Training ===
        model.train()
        total_loss = 0
        y_true_train, y_pred_train = [], []

        for seq_x, meta_x, y in train_loader:
            seq_x, meta_x, y = seq_x.to(device), meta_x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(seq_x, meta_x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            y_true_train.extend(y.detach().cpu().numpy())
            y_pred_train.extend(preds.detach().cpu().numpy())

        avg_train_loss = total_loss / len(train_loader)
        end_epoch_time = time.time()
        elapsed_train = round(end_epoch_time - start_epoch_time, 4)

        # === Validation ===
        model.eval()
        val_loss = 0
        y_true_val, y_pred_val = [], []
        
        start_epoch_time = time.time()
        with torch.no_grad():
            for seq_x, meta_x, y in val_loader:
                seq_x, meta_x, y = seq_x.to(device), meta_x.to(device), y.to(device)
                preds = model(seq_x, meta_x)
                val_loss += criterion(preds, y).item()
                y_true_val.extend(y.cpu().numpy())
                y_pred_val.extend(preds.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        end_epoch_time = time.time()
        elapsed_val = round(end_epoch_time - start_epoch_time, 4)
        

        # === Logging ===
        log_loss_values(
            model_name=model_name,
            epoch=epoch,
            start_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_epoch_time)),
            end_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_epoch_time)),
            train_loss=avg_train_loss,
            val_loss=avg_val_loss
        )

        log_training_metrics(
            model_name=model_name,
            epoch=epoch,
            elapsed_time=elapsed_train,
            y_true=y_true_train,
            y_pred=y_pred_train,
            scaler=scaler,
            type='train'
        )

        log_training_metrics(
            model_name=model_name,
            epoch=epoch,
            elapsed_time=elapsed_val,
            y_true=y_true_val,
            y_pred=y_pred_val,
            scaler=scaler,
            type='val'
        )

        print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Early stopping (optional)
        # early_stopper(avg_val_loss)
        # if early_stopper.early_stop:
        #     print("⏹️ Early stopping triggered.")
        #     break



def search_powernet_hyperparams(train_loader, val_loader, scaler, input_size_lstm, input_size_meta, horizon, epochs):
    if input_size_meta <= 0:
        raise ValueError(f"Invalid input_size_meta={input_size_meta}. Must be positive.")

    lstm_hidden_options = [64, 128, 256, 512]
    mlp_hidden1_options = [32, 64]
    mlp_hidden2_options = [16, 32]
    final_hidden_options = [32, 64]
    dropout_options = [0.2, 0.3]
    lr_options = [1e-3, 5e-4]

    # lstm_hidden_options = [64]
    # mlp_hidden1_options = [32]
    # mlp_hidden2_options = [16]
    # final_hidden_options = [512]
    # dropout_options = [0.2]
    # lr_options = [1e-3, 5e-4]

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
            horizon=horizon,
            input_size_lstm=input_size_lstm,
            lstm_hidden=lstm_h,
            lstm_layers=2,
            mlp_hidden1=mlp_h1,
            mlp_hidden2=mlp_h2,
            final_hidden=final_h,
            dropout=dropout
        )

        model_name =  f"models/powernet_lstm{lstm_h}_mlp1{mlp_h1}_mlp2{mlp_h2}_final{final_h}_dropout{dropout}_lr{lr}.pt"
        train_powernet(model, model_name, scaler, train_loader, val_loader, epochs=epochs, lr=lr, device=DEVICE)
        torch.save(model, model_name)

        model.eval()
        total_val_loss = 0
        criterion = nn.MSELoss()
        with torch.no_grad():
            for seq_x, meta_x, y in val_loader:
                seq_x, meta_x, y = seq_x.to(DEVICE), meta_x.to(DEVICE), y.to(DEVICE)
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

if __name__ == "__main__":
    data = create_calendar_features(pd.read_csv('data/mm79158.csv'), 'ts')
    wdf = preprocess_weather_data('data/slovenia_weather_averaged.csv')
    df = join_calendar_and_weather(data, wdf, 'ts')

    X_train, y_train, X_val, y_val, X_test, y_test, feature_scaler, target_scaler = prepare_powernet_data(df, test_size=0.3, val_size=0.1)

    lookback = 24 * 14
    horizon = 1
    batch_size=64
    epochs = 20

    seq_train, meta_train, y_train = build_sequence_data(X_train, y_train, lookback, horizon)
    seq_val, meta_val, y_val = build_sequence_data(X_val, y_val, lookback, horizon)

    train_loader = DataLoader(TensorDataset(seq_train, meta_train, y_train), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(seq_val, meta_val, y_val), batch_size=batch_size)

    input_size_lstm = seq_train.shape[2]   # this is 1 for univariate
    input_size_meta = meta_train.shape[1]  # this is correct size of meta features

    best_model, best_config = search_powernet_hyperparams(train_loader, val_loader, target_scaler, input_size_lstm, input_size_meta, horizon, epochs)