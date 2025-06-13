#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""

Time Series Forecasting using JITrans Model

This source code is related to the paper "Just In Time Transformers", by 
AHMED ALA EDDINE BENALI, MASSIMO CAFARO, ITALO EPICOCO, MARCO PULIMENO and ENRICO JUNIOR SCHIOPPA

This script loads the data, preprocesses it, creates sequences, defines the model,
trains the model, evaluates it, and generates plots. 

Credits:

AHMED ALA EDDINE BENALI wrote the original source code
MASSIMO CAFARO and ITALO EPICOCO revised the source code

In order to train the model, the use of a recent NVIDIA gpu 
is highly recommended to speedup the training process

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
import os
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.ticker as mticker
from collections import defaultdict
# Import define_model from model.py.
from model import define_model
from model import Transformer

# Set the random seed for reproducibility.
torch.manual_seed(42)
np.random.seed(42)

def load_and_preprocess_data(directory_path):
    """
    Loads and preprocesses time series data, applies smoothing and outlier detection.
    """
    
    # Load data from parquet, apply transformations, and generate additional features.
    data = pd.read_parquet(directory_path)
    data = data.sum(axis=1).to_frame(name='cntr')
    data.rename_axis('Date_Time', inplace=True)
    daily_data = data.resample('D').sum().reset_index()
    daily_data['cntr'] = daily_data['cntr'].round(2)
    
    #Simple moving average and outlier handling.
    window_size = 7
    daily_data['SMA_7'] = daily_data['cntr'].rolling(window=window_size).mean()
    daily_data['SMA_7'] = daily_data['SMA_7'].bfill()
    Q1 = daily_data['cntr'].quantile(0.25)
    Q3 = daily_data['cntr'].quantile(0.75)
    IQR = Q3 - Q1
    outliers_iqr = daily_data[(daily_data['cntr'] < (Q1 - 1.5 * IQR)) | (daily_data['cntr'] > (Q3 + 1.5 * IQR))]
    average_max = daily_data['cntr'].nlargest(window_size).mean()
    daily_data.loc[outliers_iqr.index, 'cntr'] = average_max
    
    # Add contextual features and scale data.
    daily_data['day_of_week'] = daily_data['Date_Time'].dt.dayofweek
    daily_data['month'] = daily_data['Date_Time'].dt.month
    daily_data['year'] = daily_data['Date_Time'].dt.year
    scaler = RobustScaler()
    daily_data['SMA_7'] = scaler.fit_transform(daily_data[['SMA_7']])
    joblib.dump(scaler, 'scalers/scaler.pkl')# Save scaler for future use.
    day_of_week_scaler = StandardScaler()
    daily_data['day_of_week'] = day_of_week_scaler.fit_transform(daily_data[['day_of_week']])
    
    # Selecting features for the model.
    features = ['SMA_7', 'day_of_week']
    target_feature = 'SMA_7'
    context_features = ['SMA_7', 'day_of_week', 'month']
    
    # Scale context data.
    context_scaler = StandardScaler()
    context_data = context_scaler.fit_transform(daily_data[context_features])
    
    # Reduce context dimensionality using PCA.
    pca = PCA(n_components=1)
    context_data_reduced = pca.fit_transform(context_data)
    
    # Prepare final data.
    final_data = daily_data[features]
    final_data_context = context_data_reduced

    return final_data, daily_data, final_data_context, scaler, target_feature

def create_encoder_decoder_sequences(input_data, target_data, context_data, encoder_seq_length, decoder_seq_length):
    """
    Generates encoder-decoder sequences for the model.
    """ 
      
    encoder_inputs, decoder_inputs, decoder_targets = [], [], []
    for i in range(len(input_data) - encoder_seq_length - decoder_seq_length):  # To ensure we don't go beyond available data.
        encoder_input = input_data[i: i + encoder_seq_length].values
        context_input = np.tile(context_data[i: i + encoder_seq_length], (1, 1))
        encoder_input = np.concatenate([encoder_input, context_input], axis=1)
        decoder_input = target_data[i + encoder_seq_length - 1: i + encoder_seq_length + decoder_seq_length - 1].values.reshape(-1, 1)
        decoder_target = target_data[i + encoder_seq_length: i + encoder_seq_length + decoder_seq_length].values.reshape(-1, 1)
        encoder_inputs.append(encoder_input)
        decoder_inputs.append(decoder_input)
        decoder_targets.append(decoder_target)
    encoder_inputs = np.array(encoder_inputs)
    decoder_inputs = np.array(decoder_inputs)
    decoder_targets = np.array(decoder_targets)

    return encoder_inputs, decoder_inputs, decoder_targets


def create_sequences(final_data, daily_data, final_data_context, encoder_seq_length, decoder_seq_length, target_feature):
    """
    Wrapper to create input/output sequences for training/testing.
    """
    
    encoder_inputs, decoder_inputs, decoder_targets = create_encoder_decoder_sequences(
        final_data, daily_data[[target_feature]], final_data_context, encoder_seq_length, decoder_seq_length)
    return encoder_inputs, decoder_inputs, decoder_targets


def create_dataloaders(encoder_inputs, decoder_inputs, decoder_targets, batch_size):
    """
    Splits data into training, validation, and test sets, and creates dataloaders.
    """
    
    # Adjust splits and dataset creation.
    total_size = encoder_inputs.shape[0]
    train_size = int(total_size * 0.6)
    val_size = int(total_size * 0.1)
    test_size = total_size - train_size - val_size

    train_encoder_inputs = encoder_inputs[:train_size]
    train_decoder_inputs = decoder_inputs[:train_size]
    train_decoder_targets = decoder_targets[:train_size]

    val_encoder_inputs = encoder_inputs[train_size:train_size+val_size]
    val_decoder_inputs = decoder_inputs[train_size:train_size+val_size]
    val_decoder_targets = decoder_targets[train_size:train_size+val_size]

    test_encoder_inputs = encoder_inputs[train_size+val_size:]
    test_decoder_inputs = decoder_inputs[train_size+val_size:]
    test_decoder_targets = decoder_targets[train_size+val_size:]
    
    # Tensor conversion for PyTorch Dataloader.
    train_enc_inputs_tensor = torch.tensor(train_encoder_inputs).float()
    train_dec_inputs_tensor = torch.tensor(train_decoder_inputs).float()
    train_targets_tensor = torch.tensor(train_decoder_targets).float()

    val_enc_inputs_tensor = torch.tensor(val_encoder_inputs).float()
    val_dec_inputs_tensor = torch.tensor(val_decoder_inputs).float()
    val_targets_tensor = torch.tensor(val_decoder_targets).float()

    test_enc_inputs_tensor = torch.tensor(test_encoder_inputs).float()
    test_dec_inputs_tensor = torch.tensor(test_decoder_inputs).float()
    test_targets_tensor = torch.tensor(test_decoder_targets).float()
    
    # Prepare dataloaders.
    train_dataset = TensorDataset(train_enc_inputs_tensor, train_dec_inputs_tensor, train_targets_tensor)
    val_dataset = TensorDataset(val_enc_inputs_tensor, val_dec_inputs_tensor, val_targets_tensor)
    test_dataset = TensorDataset(test_enc_inputs_tensor, test_dec_inputs_tensor, test_targets_tensor)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader, test_enc_inputs_tensor, test_targets_tensor

def train_model(model, train_dataloader, val_dataloader, num_epochs=50):
    """
    Trains the model for a given number of epochs, using MSE loss and Adam optimizer.
    """
    
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
            
            #Inputs require gradients.
            enc_inputs.requires_grad_(True)
            dec_inputs.requires_grad_(True)
            optimizer.zero_grad()
            outputs = model(enc_inputs, dec_inputs)
            targets = targets.squeeze(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation pass.     
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
        
        
def evaluate_model(model, test_encoder_inputs, test_targets_tensor, scaler, horizon):
    device = model.device
    predictions_per_day = [[] for _ in range(horizon)]
    true_values_per_day = [[] for _ in range(horizon)]

    for test_index in range(len(test_encoder_inputs)):
        input_sequence = test_encoder_inputs[test_index]
        true_values = test_targets_tensor[test_index].cpu().numpy().reshape(-1)
        predicted_values = make_predictions(model, input_sequence, true_values, horizon, device)

        predicted_values = scaler.inverse_transform(predicted_values.reshape(-1, 1)).flatten()
        true_values = scaler.inverse_transform(true_values.reshape(-1, 1)).flatten()

        for i in range(horizon):
            predictions_per_day[i].append(predicted_values[i])
            true_values_per_day[i].append(true_values[i])

    all_preds = np.array(predictions_per_day).flatten()
    all_true = np.array(true_values_per_day).flatten()

    mse = mean_squared_error(all_true, all_preds)
    mae = mean_absolute_error(all_true, all_preds)
    mape = mean_absolute_percentage_error(all_true, all_preds)

    return {
        "predictions_per_day": predictions_per_day,
        "true_values_per_day": true_values_per_day,
        "mse": mse,
        "mae": mae,
        "mape": mape
    }



def make_predictions(model, encoder_input, true_values, horizon, device):
    input_tensor = encoder_input.clone().detach().float().to(device).unsqueeze(0)
    final_predictions = []

    # Initialize decoder input with known values (e.g., first few true values)
    decoder_input = true_values[:horizon].reshape(-1, 1).tolist()

    for step in range(len(decoder_input), horizon):
        previous_predictions = final_predictions.copy()
        avg_input = np.mean(previous_predictions) if previous_predictions else 0.0
        decoder_input.append([avg_input])

    dec_input_tensor = torch.tensor(decoder_input).float().to(device).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor, dec_input_tensor)

    prediction = output.squeeze(0).cpu().detach().numpy()
    return prediction


def plot_results(predictions_per_day, true_values_per_day):
    """
    Plots true vs predicted values over selected days.
    """
    
    plt.rcParams.update({'font.size': 14, 'figure.figsize': (12, 8), 'savefig.dpi': 300})
    
    # Set up a 2x2 grid for subplots with a larger figure size.
    fig, axs = plt.subplots(2, 2, figsize=(20, 18))
    # Days to plot (1, 2, 4, and 7).
    days_to_plot = [1, 2, 4, 7]
    
    # Plot for each selected day.
    for idx, day in enumerate(days_to_plot):
        ax = axs[idx // 2, idx % 2]
        ax.plot(true_values_per_day[day - 1], label='True Values', marker='o', color='blue')
        ax.plot(predictions_per_day[day - 1], label='Predicted Values', linestyle='--', marker='o', color='red')
        ax.set_title(f'Day {day}', fontsize=20)
        ax.set_xlabel('Days', fontsize=16)
        ax.set_ylabel('Value (kWh)', fontsize=16)
        ax.grid(True)
        ax.tick_params(axis='both', which='major', labelsize=14)
       
        # Set y-axis to scientific notation.
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
   
    # Adjust layout to add more space between subplots.
    plt.subplots_adjust(hspace=0.5, wspace=0.4, top=0.85)
   
    #Single legend for all subplots.
    handles, labels = axs[0, 0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='center', ncol=2, fontsize=16, bbox_to_anchor=(0.5, -0.025))
   
    # Adjust layout to make sure everything fits, allowing extra space for the legend.
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    # Save the figure.
    plt.savefig('combined_plot_of_days_prediction.png', format='eps', dpi=600, bbox_extra_artists=(legend,), bbox_inches='tight')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set parameters directly
    directory_path = 'prepared_data.parquet'
    lookback = 24
    horizon = 24
    num_epochs = 30
    batch_size = 64
    model_save_path = 'model_final.pth'

    # Load and preprocess data
    final_data, daily_data, final_data_context, scaler, target_feature = load_and_preprocess_data(directory_path)
    encoder_inputs, decoder_inputs, decoder_targets = create_sequences(
        final_data, daily_data, final_data_context, lookback, horizon, target_feature)

    train_dataloader, val_dataloader, test_dataloader, test_encoder_inputs, test_targets_tensor = create_dataloaders(
        encoder_inputs, decoder_inputs, decoder_targets, batch_size)

    model = define_model(device=device, use_checkpoint=True)
    train_model(model, train_dataloader, val_dataloader, num_epochs)
    torch.save(model.state_dict(), model_save_path)

    results = evaluate_model(model, test_encoder_inputs, test_targets_tensor, scaler, horizon)
    plot_results(results['predictions_per_day'], results['true_values_per_day'])

    metrics_df = pd.DataFrame([{
        'MSE': results['mse'],
        'MAE': results['mae'],
        'MAPE': results['mape']
    }])
    metrics_df.to_csv("results/ji_trans_metrics.csv", index=False)
    print("Metrics saved to ji_trans_metrics.csv")


if __name__ == '__main__':
    main()
