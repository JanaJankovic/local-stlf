import torch
from model import SwtForecastingModel
import csv
import ctypes
from data import prepare_data
from datetime import datetime
from util import evaluate_model

# --- DEVICE SETUP ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DEVICE= torch.device("cpu")  # Force CPU for compatibility
print(f"🚀 Using device: {DEVICE}")

LOG_PATH = "logs/training_log.csv"
METRICS_PATH = "logs/training_eval.csv"


# --- CONFIGURATION ---
args = {
    "test": 0.3,  # test/val split ratio
    "val": 0.1, 
    "horizon": 8,      # horizon=8 only due to SWT even-length requirement.
    "lookback": 24 * 14,
    "epochs": 20,  
    "batch_size": 64,
    "level": 3,  # user-defined max SWT level
    "wavelet": "db2"
}

def get_model():
    return SwtForecastingModel(
        input_size=1,
        time2vec_k=8,
        #d_model=480, # Original paper uses 480, but it is too large for this dataset
        d_model=64,
        #n_heads=12, # Original paper uses 12, but it is too large for this dataset
        n_heads=2,
        d_ff=128,
        n_enc_layers=2,
        n_dec_layers=0,
        forecast_steps=args['horizon'],
        output_bands=len(X_train_tensors)
    ).to(DEVICE)



def train_model(train_loader, val_loader, ys, scalers_y, args):
    model = get_model()

    y_train, y_val, _ = ys

    epochs = args['epochs']
    # Paper uses RMSprop, but it has given unpromising results and is quite slow.
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0)

    loss_fn = torch.nn.MSELoss()
    #loss_fn = torch.nn.SmoothL1Loss() # experimental test

    for epoch in range(epochs):
        start_time = datetime.now()
        model.train()
        train_loss = 0
        num_batches = len(train_loader)
        print(f"\nEpoch {epoch+1:02d}/{epochs}")

        for batch_idx, batch in enumerate(train_loader):
            num_bands = len(batch) // 2
            x_band_tensor = [x.to(DEVICE) for x in batch[:num_bands]]
            y_band_tensor = [y.to(DEVICE) for y in batch[num_bands:]]

            x_stack = torch.stack(x_band_tensor, dim=1)  # [B, bands, W, 1]
            y_stack = torch.stack(y_band_tensor, dim=1)  # [B, bands, horizon, 1]

            preds = model(x_stack)
            loss = loss_fn(preds, y_stack)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            #scheduler.step()

            train_loss += loss.item() * y_stack.size(0)

            print(f"\r - Batch {batch_idx+1}/{num_batches} - loss: {loss.item():.6f}", end='', flush=True)
        
        end_time = datetime.now()
        
        _, train_metrics = evaluate_model(model, train_loader, y_train, scalers_y[0], args['horizon'], data_type='train', wavelet='db2')
        train_loss /= len(train_loader.dataset)

        # === Validation ===
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                num_bands = len(batch) // 2
                x_band_tensor = [x.to(DEVICE) for x in batch[:num_bands]]
                y_band_tensor = [y.to(DEVICE) for y in batch[num_bands:]]

                x_stack = torch.stack(x_band_tensor, dim=1)
                y_stack = torch.stack(y_band_tensor, dim=1)

                preds = model(x_stack)
                loss = loss_fn(preds, y_stack)
                val_loss += loss.item() * y_stack.size(0)

        val_loss /= len(val_loader.dataset)
        _, val_metrics = evaluate_model(model, val_loader, y_val, scalers_y[0], args['horizon'], data_type='val', wavelet='db2')
        print(f"\n[Epoch {epoch+1:02d}] Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f}")

        
        with open(LOG_PATH, 'a', newline='') as f:
            csv.writer(f).writerow([
                start_time.strftime('%Y-%m-%d %H:%M:%S'),
                end_time.strftime('%Y-%m-%d %H:%M:%S'),
                epoch + 1,
                train_loss,
                val_loss
            ])

        with open(METRICS_PATH, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "type", *train_metrics.keys()])
            writer.writerow({"epoch": epoch + 1, "type": "train", **train_metrics})
            writer.writerow({"epoch": epoch + 1, "type": "val", **val_metrics})

        torch.save(model, f"models/model_epoch_{epoch + 1}.pt")
    print("✅ Model saved.")



if __name__ == "__main__":

    # Prevent sleep
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        # --- CONFIG ---

    with open(LOG_PATH, 'w', newline='') as f:
        csv.writer(f).writerow(['start_epoch_time', 'end_epoch_time', 'epoch', 'train_loss', 'val_loss'])

    with open(METRICS_PATH, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch','type','inference','MAE','MSE','RMSE','MAPE','R2','MDA','Spearman'])

    train_loader, val_loader, test_loader, ys, scalers_X, scalers_y, X_train_tensors, X_val_tensors, X_test_tensors = prepare_data(args)
    train_model(train_loader, val_loader, ys, scalers_y, args)
    print("Training complete. Model saved as 'models/model.pth'.")

    # Allow sleep again
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)