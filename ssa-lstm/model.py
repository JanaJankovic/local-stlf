import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.extmath import randomized_svd

class LoadIntervalForecaster:
    class SSALSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])  # output last timestep

    class RegularSubsequenceDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            return self.X[i], self.y[i]

    def __init__(self, input_dim, hidden_dim=64, num_layers=2, sigma=3, batch_size=32, lr=0.001):
        self.model = self.SSALSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=1)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.sigma = sigma
        self.batch_size = batch_size
        self.df_to_std_map = None

    def ssa_decompose(self, series, window=24, trend_components=2, periodic_components=3):
        """
        SSA to split into trend, periodic, and stochastic (noise) subsequences.

        Args:
            series (np.ndarray): 1D time series
            window (int): window size for SSA
            trend_components (int): number of components for trend
            periodic_components (int): number of components for periodic

        Returns:
            trend_series (np.ndarray): reconstructed trend
            periodic_series (np.ndarray): reconstructed periodic pattern
            stochastic_series (np.ndarray): residual noise
        """
        N = len(series)
        K = N - window + 1

        # === Step 1: Trajectory matrix (Hankel embedding) ===
        X = np.column_stack([series[i:i + window] for i in range(K)])

        # === Step 2: SVD ===
        from sklearn.utils.extmath import randomized_svd
        U, S, Vt = randomized_svd(X, n_components=window, random_state=42)

        # === Step 3: Group components ===
        trend = (S[:trend_components, None] * U[:, :trend_components]) @ Vt[:trend_components, :]
        periodic = (S[trend_components:trend_components + periodic_components, None] *
                    U[:, trend_components:trend_components + periodic_components]) @ Vt[trend_components:trend_components + periodic_components, :]
        stochastic = (S[trend_components + periodic_components:, None] *
                    U[:, trend_components + periodic_components:]) @ Vt[trend_components + periodic_components:, :]

        # === Step 4: Diagonal averaging ===
        def diagonal_avg(matrix):
            L, K = matrix.shape
            result = np.zeros(L + K - 1)
            count = np.zeros(L + K - 1)
            for i in range(L):
                for j in range(K):
                    result[i + j] += matrix[i, j]
                    count[i + j] += 1
            return result / count

        trend_series = diagonal_avg(trend)
        periodic_series = diagonal_avg(periodic)
        stochastic_series = diagonal_avg(stochastic)

        return trend_series, periodic_series, stochastic_series
    
    
    def prepare_data(self, X_regular, y):
        dataset = self.RegularSubsequenceDataset(X_regular, y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def train(self, X_regular, y, epochs=50):
        loader = self.prepare_data(X_regular, y)
        self.model.train()
        for epoch in range(epochs):
            for X_batch, y_batch in loader:
                self.optimizer.zero_grad()
                preds = self.model(X_batch).squeeze()
                loss = self.criterion(preds, y_batch)
                loss.backward()
                self.optimizer.step()
            print(f"[Epoch {epoch+1}] Loss: {loss.item():.4f}")

    def set_df_to_std_map(self, df_std_mapping):
        self.df_to_std_map = df_std_mapping

    def forecast_with_intervals(self, X_regular, df_values):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(torch.tensor(X_regular, dtype=torch.float32)).squeeze().numpy()
        stds = np.array([self.df_to_std_map.get(round(df, 1), 1.0) for df in df_values])
        lower = preds - self.sigma * stds
        upper = preds + self.sigma * stds
        return preds, lower, upper
