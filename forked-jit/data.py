import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch
import joblib


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
    train_idx, val_idx, test_idx = split_by_index(all_target_indices)

    def make_loader(enc, dec, tgt):
        dataset = TensorDataset(torch.tensor(enc).float(), torch.tensor(dec).float(), torch.tensor(tgt).float())
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return (
        make_loader(train_enc, train_dec, train_tgt),
        make_loader(val_enc, val_dec, val_tgt),
        make_loader(test_enc, test_dec, test_tgt),
        (train_idx, val_idx, test_idx),
        df
    )