import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class LoadDataset(Dataset):
    def __init__(self, df, horizon=24):
        self.horizon = horizon
        features = ["time", "day_of_year", "day_type"]
        self.X = torch.tensor(df[features].values, dtype=torch.float32)
        y = df["vrednost"].values
        self.y = torch.from_numpy(np.stack([y[i:i+horizon] for i in range(len(y)-horizon)])).float()
        self.task_ids = torch.tensor(df["task_id"].values[:len(self.y)], dtype=torch.long)
        self.X = self.X[:len(self.y)]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.task_ids[idx]

def load_and_preprocess(path, k=2):
    df = pd.read_csv(path, parse_dates=["ts"])

    # Convert wide to long format
    df_long = df.melt(id_vars=["ts"], var_name="task_id", value_name="vrednost")
    df_long.dropna(subset=["vrednost"], inplace=True)

    # Simplify task_id to numeric labels
    df_long["task_id"] = df_long["task_id"].astype("category").cat.codes

    # Filter to first k task_ids
    selected_task_ids = df_long["task_id"].unique()[:k]
    df_long = df_long[df_long["task_id"].isin(selected_task_ids)]

    # Round all timestamps to start of the hour BEFORE any trimming or aggregation
    df_long["ts"] = df_long["ts"].dt.floor("h")

    # Group and aggregate hourly values per task_id
    df_agg = df_long.groupby(["task_id", "ts"], as_index=False)["vrednost"].mean()

    # Add time-based features after aggregation
    df_agg["time"] = df_agg["ts"].dt.hour
    df_agg["day_of_year"] = df_agg["ts"].dt.dayofyear
    df_agg["day_type"] = df_agg["ts"].dt.weekday

    # Final columns
    df_final = df_agg[["time", "day_of_year", "day_type", "vrednost", "task_id"]].copy()
    return df_final



def split_by_task(df, val_ratio=0.2, test_ratio=0.2):
    train_parts, val_parts, test_parts = [], [], []
    scalers = {}

    for task_id in df["task_id"].unique():
        df_task = df[df["task_id"] == task_id].copy()
        total_len = len(df_task)

        test_size = int(total_len * test_ratio)
        val_size = int(total_len * val_ratio)
        train_size = total_len - test_size - val_size

        train = df_task.iloc[:train_size].copy()
        val = df_task.iloc[train_size:train_size + val_size].copy()
        test = df_task.iloc[train_size + val_size:].copy()

        feature_cols = ["time", "day_of_year", "day_type"]
        feature_scaler = MinMaxScaler()
        feature_scaler.fit(train[feature_cols])

        target_scaler = MinMaxScaler()
        target_scaler.fit(train[["vrednost"]])

        for split in [train, val, test]:
            split[feature_cols] = feature_scaler.transform(split[feature_cols])
            split[["vrednost"]] = target_scaler.transform(split[["vrednost"]])

        scalers[task_id] = {
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
        }

        train_parts.append(train)
        val_parts.append(val)
        test_parts.append(test)

    return (
        pd.concat(train_parts).reset_index(drop=True),
        pd.concat(val_parts).reset_index(drop=True),
        pd.concat(test_parts).reset_index(drop=True),
        scalers
    )