import torch
import torch.nn as nn
from model import AutoformerForecast
import torch.optim as optim
import csv
import time
import ctypes
from data import prepare_data
from util import log_metrics_per_epoch
import os
from datetime import timedelta
import sys

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_losses = []
    num_batches = len(dataloader)
    start_time = time.time()

    for batch_idx, (x_load, x_weather_hist, x_weather_fore, y) in enumerate(dataloader):
        x_load = x_load.to(device)
        x_weather_hist = x_weather_hist.to(device)
        x_weather_fore = x_weather_fore.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        output = model(x_load, x_weather_hist, x_weather_fore)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

        # === ⏱ Logging ===
        elapsed = time.time() - start_time
        avg_batch_time = elapsed / (batch_idx + 1)
        remaining = avg_batch_time * (num_batches - batch_idx - 1)

        sys.stdout.write(
            f"\rBatch {batch_idx+1}/{num_batches} | "
            f"Loss: {loss.item():.4f} | "
            f"Elapsed: {timedelta(seconds=int(elapsed))} | "
            f"ETA: {timedelta(seconds=int(remaining))}"
        )
        sys.stdout.flush()

    print()  # new line after epoch
    return sum(epoch_losses) / len(epoch_losses)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    val_losses = []

    with torch.no_grad():
        for x_load, x_weather_hist, x_weather_fore, y in dataloader:
            x_load = x_load.to(device)
            x_weather_hist = x_weather_hist.to(device)
            x_weather_fore = x_weather_fore.to(device)
            y = y.to(device)

            output = model(x_load, x_weather_hist, x_weather_fore)
            loss = criterion(output, y)
            val_losses.append(loss.item())

    return sum(val_losses) / len(val_losses)


def train_model(model, train_loader, val_loader, scaler, num_epochs=50, patience=5,
                lr=1e-3, save_path='models/model.pt', log_path='logs/training_log.csv',
                eval_log_path='logs/training_eval.csv'):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    patience_counter = 0

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, mode='w', newline='') as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["start_epoch_time", "end_epoch_time", "epoch", "train_loss", "val_loss"])

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        start_time = time.strftime("%Y-%m-%d %H:%M:%S")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        end_time = time.strftime("%Y-%m-%d %H:%M:%S")

        with open(log_path, mode='a', newline='') as log_file:
            writer = csv.writer(log_file)
            writer.writerow([start_time, end_time, epoch+1, train_loss, val_loss])

        print(f"Epoch {epoch+1} Summary | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        log_metrics_per_epoch(model, epoch, train_loader, val_loader, device, eval_log_path, scaler)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model, save_path)
            print("  ✅ New best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("  ⏹️ Early stopping triggered.")
                break

    return torch.load(save_path)

if __name__ == '__main__':

    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

    csv_path = 'data/mm79158.csv'
    seq_len = 24 * 14
    horizon = 1
    batch_size = 64
    train_loader, val_loader, test_loader, scaler = prepare_data(csv_path, seq_len, horizon, batch_size=batch_size)
    model = AutoformerForecast(d_model=32, kernel_size=24, top_k=3, horizon=horizon)
    trained_model = train_model(model, train_loader, val_loader, scaler, num_epochs=20, patience=20)

    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)



 