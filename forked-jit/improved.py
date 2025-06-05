import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import joblib
import torch
from torch.utils.data import TensorDataset, DataLoader
from model import define_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt

def create_encoder_decoder_sequences(input_data, target_data, context_data, encoder_seq_length, decoder_seq_length):
    encoder_inputs, decoder_inputs, decoder_targets, decoder_target_indices = [], [], [], []
    total_len = len(input_data)

    for i in range(total_len - encoder_seq_length - decoder_seq_length + 1):
        encoder_input = input_data.iloc[i: i + encoder_seq_length].values
        context_input = np.tile(context_data[i: i + encoder_seq_length], (1, 1))
        encoder_input = np.concatenate([encoder_input, context_input], axis=1)

        decoder_input = target_data[i + encoder_seq_length - 1: i + encoder_seq_length + decoder_seq_length - 1].reshape(-1, 1)
        decoder_target = target_data[i + encoder_seq_length: i + encoder_seq_length + decoder_seq_length].reshape(-1, 1)

        encoder_inputs.append(encoder_input)
        decoder_inputs.append(decoder_input)
        decoder_targets.append(decoder_target)
        decoder_target_indices.append(np.arange(i + encoder_seq_length, i + encoder_seq_length + decoder_seq_length))

    return (
        np.array(encoder_inputs),
        np.array(decoder_inputs),
        np.array(decoder_targets),
        np.array(decoder_target_indices)
    )

def align_forecast_to_indices(y_pred, index_windows, total_len):
    aligned = np.full(total_len, np.nan)
    for window, idx_seq in zip(y_pred, index_windows):
        for val, idx in zip(window, idx_seq):
            aligned[idx] = val
    return aligned

def feature_engineering(csv_path, window_size=24):
    df = pd.read_csv(csv_path, sep=";", decimal=",", parse_dates=["ts"])
    df = df.set_index("ts").rename(columns={'vrednost': 'cntr'})
    df.rename_axis('Date_Time', inplace=True)
    df = df.resample('h').sum().reset_index()
    df['cntr'] = df['cntr'].round(2)
    df['SMA_7'] = df['cntr'].rolling(window=window_size).mean().bfill()

    Q1, Q3 = df['cntr'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    outliers = (df['cntr'] < (Q1 - 1.5 * IQR)) | (df['cntr'] > (Q3 + 1.5 * IQR))
    average_max = df['cntr'].nlargest(window_size).mean()
    df.loc[outliers, 'cntr'] = average_max

    df['day_of_week'] = df['Date_Time'].dt.dayofweek
    df['month'] = df['Date_Time'].dt.month
    df['year'] = df['Date_Time'].dt.year

    return df

def scaling_and_pca(df, test_size=0.3, val_size=0.1):
    from sklearn.base import TransformerMixin

    def apply_scaler(scaler: TransformerMixin, source_df: pd.DataFrame, feature: str, indices: np.ndarray) -> np.ndarray:
        return scaler.transform(source_df.loc[indices, [feature]])[:, 0]

    def scale_column(df_out: pd.DataFrame, source_df: pd.DataFrame, feature: str, scaler: TransformerMixin, split_key: str):
        df_out.loc[splits[split_key], feature] = apply_scaler(scaler, source_df, feature, splits[split_key])

    total_len = len(df)
    test_len = int(total_len * test_size)
    val_len = int(total_len * val_size)
    train_len = total_len - test_len - val_len
    splits = {
        'train': np.arange(0, train_len),
        'val': np.arange(train_len, train_len + val_len),
        'test': np.arange(train_len + val_len, total_len)
    }

    raw_target = df[['SMA_7']].copy()
    raw_features = df[['SMA_7', 'day_of_week']].astype(float).copy()
    raw_context = df[['SMA_7', 'day_of_week', 'month']].astype(float).copy()

    scaler_target = MinMaxScaler().fit(raw_target.loc[splits['train']])
    scaler_day = StandardScaler().fit(raw_features.loc[splits['train'], ['day_of_week']])
    context_scaler = StandardScaler().fit(raw_context.loc[splits['train']])
    pca = PCA(n_components=1).fit(context_scaler.transform(raw_context.loc[splits['train']]))

    scaled_target = raw_target.copy()
    for split in splits:
        scale_column(scaled_target, raw_target, 'SMA_7', scaler_target, split)

    scaled_features = raw_features.copy()
    for split in splits:
        scale_column(scaled_features, raw_features, 'day_of_week', scaler_day, split)

    reduced_context = np.zeros((total_len, 1))
    for split in splits:
        reduced_context[splits[split]] = pca.transform(context_scaler.transform(raw_context.loc[splits[split]]))

    joblib.dump(scaler_target, 'scalers/scaler_target.pkl')
    joblib.dump(scaler_day, 'scalers/scaler_day.pkl')
    joblib.dump(context_scaler, 'scalers/scaler_context.pkl')
    joblib.dump(pca, 'scalers/pca.pkl')

    return scaled_features, scaled_target, reduced_context, splits['train'], splits['val'], splits['test']

def preprocess_and_split_data(csv_path, encoder_seq_length, decoder_seq_length, batch_size=32, test_size=0.3, val_size=0.1):
    df = feature_engineering(csv_path)
    df = df.iloc[encoder_seq_length + decoder_seq_length:].reset_index(drop=True)
    features, target, reduced_context, _, _, _ = scaling_and_pca(df, test_size=test_size, val_size=val_size)

    enc_inputs, dec_inputs, dec_targets, all_target_indices = create_encoder_decoder_sequences(
        features, target.values, reduced_context, encoder_seq_length, decoder_seq_length
    )

    def split_by_index(arr):
        total = len(arr)
        train_end = int(total * (1 - test_size - val_size))
        val_end = int(total * (1 - test_size))
        return arr[:train_end], arr[train_end:val_end], arr[val_end:]

    train_enc, val_enc, test_enc = split_by_index(enc_inputs)
    train_dec, val_dec, test_dec = split_by_index(dec_inputs)
    train_tgt, val_tgt, test_tgt = split_by_index(dec_targets)
    _, _, test_target_indices = split_by_index(all_target_indices)

    def make_loader(enc, dec, tgt):
        dataset = TensorDataset(torch.tensor(enc).float(), torch.tensor(dec).float(), torch.tensor(tgt).float())
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return (
        make_loader(train_enc, train_dec, train_tgt),
        make_loader(val_enc, val_dec, val_tgt),
        make_loader(test_enc, test_dec, test_tgt),
        test_target_indices,
        df
    )

def predict(model, test_loader, target_indices, total_len, device='cpu', scaler_dir='scalers'):
    scaler_target = joblib.load(f'{scaler_dir}/scaler_target.pkl')
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for enc_input, dec_input, _ in test_loader:
            enc_input = enc_input.to(device)
            dec_input = dec_input.to(device)
            output = model(enc_input, dec_input)
            all_predictions.append(output.cpu().numpy())

    y_pred_scaled = np.concatenate(all_predictions, axis=0)
    y_pred = scaler_target.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(y_pred_scaled.shape)
    pred_series = align_forecast_to_indices(y_pred, target_indices, total_len)
    return pred_series


def reconstruct_predictions(y_pred, horizon):
    """
    Reconstruct 1D predicted signal from windowed forecasts (y_pred shape: [num_windows * horizon])
    """
    y_pred = y_pred.reshape(-1, horizon)
    
    reconstructed = list(y_pred[0])  # take full first horizon
    for i in range(1, len(y_pred)):
        reconstructed.append(y_pred[i][-1])  # only append the last value of each next window

    return np.array(reconstructed)


def train_model(model, train_dataloader, val_dataloader, num_epochs=50):
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    device = model.device
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for enc_inputs, dec_inputs, targets in train_dataloader:
            enc_inputs = enc_inputs.to(device)
            dec_inputs = dec_inputs.to(device)
            targets = targets.to(device)
            enc_inputs.requires_grad_(True)
            dec_inputs.requires_grad_(True)
            optimizer.zero_grad()
            outputs = model(enc_inputs, dec_inputs)
            targets = targets.squeeze(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for enc_inputs, dec_inputs, targets in val_dataloader:
                enc_inputs = enc_inputs.to(device)
                dec_inputs = dec_inputs.to(device)
                targets = targets.to(device)
                outputs = model(enc_inputs, dec_inputs)
                targets = targets.squeeze(-1)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_dataloader):.4f}, Val Loss: {val_loss/len(val_dataloader):.4f}')


def reconstruct_predictions(y_pred, horizon):
    """
    Reconstruct 1D predicted signal from windowed forecasts (y_pred shape: [num_windows * horizon])
    """
    y_pred = y_pred.reshape(-1, horizon)
    
    reconstructed = list(y_pred[0])  # take full first horizon
    for i in range(1, len(y_pred)):
        reconstructed.append(y_pred[i][-1])  # only append the last value of each next window

    return np.array(reconstructed)

def main():
    lookback = 24
    horizon = 24
    num_epochs = 30
    batch_size = 64
    model_save_path = 'model_final.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Preprocessing and data loading
    train_dataloader, val_dataloader, test_dataloader, test_indices, df = preprocess_and_split_data(
        'mm79158.csv', lookback, horizon, batch_size=batch_size
    )

    # --- Define and train model
    model = define_model(device=device, use_checkpoint=True)
    train_model(model, train_dataloader, val_dataloader, num_epochs)
    torch.save(model, model_save_path)

    # --- Predict 
    y_pred = predict(model, test_dataloader, test_indices, len(df), device=device, scaler_dir='scalers')
    y_true = df['SMA_7'].values

    # Plot
    plt.figure(figsize=(30, 7))
    plt.plot(y_true, label="True Values", color="blue", linewidth=0.5)
    plt.plot(y_pred, label="Predicted Values", color="red", linewidth=0.5)
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.title("Ground Truth vs Predicted Signal")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot/true_vs_predicted.png", dpi=300, bbox_inches='tight')

    # --- Evaluate and save metrics
    y_true_1 = y_true[int(len(df)*0.7):]
    y_pred_1 = y_pred[int(len(df)*0.7):]

    mse = mean_squared_error(y_true_1, y_pred_1)
    mae = mean_absolute_error(y_true_1, y_pred_1)
    mape = mean_absolute_percentage_error(y_true_1, y_pred_1)

    pd.DataFrame({
        'mse': [mse],
        'mae': [mae],
        'mape': [mape]
    }).to_csv('fixed_metrics.csv', index=False)

if __name__ == '__main__':
    main()
