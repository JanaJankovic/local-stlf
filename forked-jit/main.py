import torch
from model import define_model
import torch
from torch import nn
import torch.optim as optim
from data import preprocess_and_split_data
import csv
from datetime import datetime
from util import evaluate_model

LOG_PATH = "logs/training_log.csv"
METRICS_PATH = "logs/training_eval.csv"


def train_model(model, train_dataloader, val_dataloader, idx, df, num_epochs=50):
    train_idx, val_idx, _ = idx

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    device = model.device
    model.to(device)
    num_batches = len(train_dataloader)

    for epoch in range(num_epochs):
        start_time = datetime.now()
        model.train()
        train_loss = 0.0
        print(f"\nEpoch {epoch+1:02d}/{num_epochs}")
        for batch_idx, (enc_inputs, dec_inputs, targets) in enumerate(train_dataloader):
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

            print(f"\r - Batch {batch_idx+1}/{num_batches} - loss: {loss.item():.6f}", end='', flush=True)

        end_time = datetime.now()
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

        print(f"\n[Epoch {epoch+1:02d}] Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f}")
        
        with open(LOG_PATH, 'a', newline='') as f:
            csv.writer(f).writerow([
                start_time.strftime('%Y-%m-%d %H:%M:%S'),
                end_time.strftime('%Y-%m-%d %H:%M:%S'),
                epoch + 1,
                train_loss,
                val_loss
            ])

        _, _, train_metrics = evaluate_model(model, train_dataloader, train_idx, df, 'train')
        _, _, val_metrics = evaluate_model(model, val_dataloader, val_idx, df, 'val')
        
        with open(METRICS_PATH, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "type", *train_metrics.keys()])
            writer.writerow({"epoch": epoch + 1, "type": "train", **train_metrics})
            writer.writerow({"epoch": epoch + 1, "type": "val", **val_metrics})


        torch.save(model, f'models/model_epoch_{epoch+1}.pt')
        


if __name__ == '__main__':
    lookback = 24 * 14
    horizon = 1
    num_epochs = 20
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    with open(LOG_PATH, 'w', newline='') as f:
        csv.writer(f).writerow(['start_epoch_time', 'end_epoch_time', 'epoch', 'train_loss', 'val_loss'])

    with open(METRICS_PATH, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch','type','inference','MAE','MSE','RMSE','MAPE','R2','MDA','Spearman'])


    # --- Preprocessing and data loading
    train_dataloader, val_dataloader, test_dataloader, idx, df = preprocess_and_split_data(
        'data/mm79158.csv', lookback, horizon, batch_size=batch_size
    )

    # --- Define and train model
    model = define_model(device=device, use_checkpoint=False)
    train_model(model, train_dataloader, val_dataloader, idx, df, num_epochs)
