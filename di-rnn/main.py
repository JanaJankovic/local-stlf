import torch
from util import RMSELoss, log_full_model_metrics_per_epoch, log_training_loss
from data import preprocess_data
from model import DIRNN
import time

def freeze_all_except(model, component_name):
    """Freeze all parameters except the specified component."""
    for param in model.parameters():
        param.requires_grad = False
    component = getattr(model, component_name)
    for param in component.parameters():
        param.requires_grad = True
    return component


def run_forward(model, component_name, x_seq, x_per):
    """Run forward pass depending on which component is being trained."""
    if component_name == 's_rnn':
        return model.s_rnn(x_seq)
    elif component_name == 'p_rnn':
        return model.p_rnn(x_per)
    else:
        return model(x_seq, x_per)


def train_one_epoch(model, component_name, x_seq, x_per, y_true, optimizer, criterion):
    """Perform one training step."""
    model.train()
    optimizer.zero_grad()

    pred = run_forward(model, component_name, x_seq, x_per)
    loss = criterion(pred, y_true)
    loss.backward()
    optimizer.step()

    return loss.item()


def validate_one_epoch(model, component_name, val_data, criterion):
    """Evaluate model on validation data."""
    model.eval()
    with torch.no_grad():
        x_seq_val, x_per_val, y_val = val_data
        val_pred = run_forward(model, component_name, x_seq_val, x_per_val)
        val_loss = criterion(val_pred, y_val)
    return val_loss.item()


def train_component(model, component_name, x_seq, x_per, y_true, val_data,
                    scaler, epochs, lr, device, loss_log_path,
                    patience=5, min_epochs=10):
    print(f"\n🧠 Training {component_name}...")
    criterion = RMSELoss()
    component = freeze_all_except(model, component_name)
    optimizer = torch.optim.Adam(component.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train_one_epoch(model, component_name, x_seq, x_per, y_true, optimizer, criterion)
        val_loss = validate_one_epoch(model, component_name, val_data, criterion)

        end_time = time.time()
        print(f"\r📘 [{component_name}] Epoch {epoch+1}/{epochs} — Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}", end="\r")

        # Log training + validation loss per epoch
        log_training_loss(loss_log_path, epoch, train_loss, val_loss, start_time, end_time)

        if epoch < min_epochs:
            continue

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_model_state = component.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️ Early stopping {component_name} at epoch {epoch+1}. Best Val Loss: {best_val_loss:.6f}")
                break

    if best_model_state:
        component.load_state_dict(best_model_state)

    # Log evaluation metrics for the full model (only for bpnn)
    if component_name == 'bpnn':
        log_full_model_metrics_per_epoch(model, x_seq, x_per, y_true, val_data, scaler, epoch, eval_log_path="logs/train_eval.csv")


def train_dirnn(model, train_data, val_data, epochs=20, lr_rnn=0.005, lr_bpnn=0.008, device='cpu', patience=5, min_epochs=10):
    print("🚂 Starting full DI-RNN training...")
    model = model.to(device)

    X_seq_train, X_per_train, y_train = [torch.tensor(x, dtype=torch.float32).to(device) for x in train_data]
    X_seq_val, X_per_val, y_val = [torch.tensor(x, dtype=torch.float32).to(device) for x in val_data]

    print("Train target shape:", y_train.shape)
    print("Train pred shape:  ", model(X_seq_train, X_per_train).shape)

    val_tensors = (X_seq_val, X_per_val, y_val)

    # === Train each component ===
    train_component(
        model, 's_rnn',
        X_seq_train, X_per_train, y_train, val_tensors,
        scaler=scaler,
        epochs=epochs, lr=lr_rnn, device=device,
        patience=patience, min_epochs=min_epochs,
        loss_log_path='logs/srnn_train_log.csv'
    )

    train_component(
        model, 'p_rnn',
        X_seq_train, X_per_train, y_train, val_tensors,
        scaler=scaler,
        epochs=epochs, lr=lr_rnn, device=device,
        patience=patience, min_epochs=min_epochs,
        loss_log_path='logs/prnn_train_log.csv'
    )

    train_component(
        model, 'bpnn',
        X_seq_train, X_per_train, y_train, val_tensors,
        scaler=scaler,
        epochs=epochs, lr=lr_bpnn, device=device,
        patience=patience, min_epochs=min_epochs,
        loss_log_path='logs/bpnn_train_log.csv'
    )
    return model

if __name__ == "__main__":
    device = "cpu"
    csv_path = 'data/mm79158.csv'
    m = 24 * 14
    n = 7
    horizon = 1
    epochs = 20

    data, scaler, _ = preprocess_data(csv_path, m=m, n=n, freq='1h', horizon=1)

    train_data = data['train']
    val_data = data['val']
    test_data = data['test']

    print("🧠 Initializing DIRNN model...")
    model = DIRNN(seq_input_size=1, per_input_size=1, hidden_size=64, bp_hidden_size=128, dropout=0.2, horizon=horizon)
    train_dirnn(model, train_data, val_data, epochs=epochs, lr_rnn=0.005, lr_bpnn=0.008, patience=3, device=device, min_epochs=epochs)

    torch.save(model, f"models/dirnn_model_{m}_{n}.pth")