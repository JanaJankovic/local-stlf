import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class LoadDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)
        # If you want task_id separately:
        self.task_ids = torch.tensor(X["task_id"].values, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Return features, target, task_id
        # If you want to exclude 'task_id' from features, do it here:
        features_only = torch.cat(
            [self.X[idx, 1:].unsqueeze(0)], dim=0  # assuming task_id is at X[:,0]
        ).squeeze(0)
        return features_only, self.y[idx], self.task_ids[idx]


def load_and_preprocess(path, selected_task_names=None):
    df = pd.read_csv(path, parse_dates=["ts"])

    # Get task column names (skip 'ts')
    task_columns = [col for col in df.columns if col != "ts"]

    # If task names are provided, use them; else, use all
    if selected_task_names is not None:
        selected_columns = ["ts"] + [
            col for col in task_columns if col in selected_task_names
        ]
    else:
        selected_columns = ["ts"] + task_columns

    df = df[selected_columns]

    # Wide to long format
    df_long = df.melt(id_vars=["ts"], var_name="task_id", value_name="vrednost")
    df_long.dropna(subset=["vrednost"], inplace=True)

    # Encode task_id as integer
    df_long["task_id"] = df_long["task_id"].astype("category").cat.codes

    # Round all timestamps to start of the hour
    df_long["ts"] = df_long["ts"].dt.floor("h")

    # Aggregate hourly values per task_id
    df_agg = df_long.groupby(["task_id", "ts"], as_index=False)["vrednost"].mean()

    # Add time-based features
    df_agg["time"] = df_agg["ts"].dt.hour
    df_agg["day_of_year"] = df_agg["ts"].dt.dayofyear
    df_agg["day_type"] = df_agg["ts"].dt.weekday

    # X: features incl. task_id; y: load
    feature_cols = ["task_id", "time", "day_of_year", "day_type"]
    X = df_agg[feature_cols].copy()
    y = df_agg["vrednost"].copy()
    return X, y


def split_by_task(X, y, val_ratio=0.2, test_ratio=0.1):
    # X and y are DataFrames/Series (same order), X has column 'task_id'
    train_idx, val_idx, test_idx = [], [], []

    for task_id in X["task_id"].unique():
        idx = X.index[X["task_id"] == task_id].to_numpy()
        n = len(idx)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        n_train = n - n_val - n_test

        # Ensure no overlap
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train : n_train + n_val])
        test_idx.extend(idx[n_train + n_val :])

    return (
        X.loc[train_idx].reset_index(drop=True),
        y.loc[train_idx].reset_index(drop=True),
        X.loc[val_idx].reset_index(drop=True),
        y.loc[val_idx].reset_index(drop=True),
        X.loc[test_idx].reset_index(drop=True),
        y.loc[test_idx].reset_index(drop=True),
    )


def merge_all_dfs():
    import os

    directory = "data"
    min_length = 80_000
    valid_dfs = {}

    # Step 1: Load and filter files with enough data
    for filename in os.listdir(directory):
        if filename.startswith("mm") and filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            try:
                df = pd.read_csv(
                    file_path, usecols=["ts", "vrednost"], parse_dates=["ts"]
                )
                df = df.sort_values("ts")
                df["vrednost"] = df["vrednost"].interpolate(
                    method="linear", limit_direction="both"
                )
                df = df.dropna(subset=["vrednost"])

                if len(df) >= min_length:
                    df = df.rename(columns={"vrednost": filename})
                    valid_dfs[filename] = df
                    print(f"✅ Loaded {filename} with {len(df)} rows")
                else:
                    print(f"⚠️ Skipped {filename} (only {len(df)} rows)")
            except Exception as e:
                print(f"❌ Error reading {filename}: {e}")

    # Step 2: Determine common timestamp range
    if not valid_dfs:
        print("❌ No datasets with sufficient length found.")
        exit()

    start_times = [df["ts"].iloc[0] for df in valid_dfs.values()]
    end_times = [df["ts"].iloc[-1] for df in valid_dfs.values()]
    common_start = max(start_times)
    common_end = min(end_times)

    # Step 3: Trim and reindex each DataFrame
    for name in list(valid_dfs):
        df = valid_dfs[name]
        trimmed = df[(df["ts"] >= common_start) & (df["ts"] <= common_end)].copy()

        if len(trimmed) < min_length:
            print(
                f"⚠️ Excluded {name} after trimming to common range ({len(trimmed)} rows)"
            )
            del valid_dfs[name]
        else:
            # Fill missing timestamps hourly only for this DF
            full_index = pd.date_range(start=common_start, end=common_end, freq="h")
            trimmed = trimmed.set_index("ts").reindex(full_index)
            trimmed.index.name = "ts"
            trimmed[name] = trimmed[name].interpolate(
                method="linear", limit_direction="both"
            )
            valid_dfs[name] = trimmed.reset_index()
            print(
                f"✅ Trimmed & filled {name} to common hourly range: {len(trimmed)} rows"
            )

    # Step 4: Merge on 'ts'
    if valid_dfs:
        merged_df = list(valid_dfs.values())[0]
        for name, df in list(valid_dfs.items())[1:]:
            merged_df = pd.merge(merged_df, df, on="ts", how="inner")

        merged_df = merged_df.sort_values("ts")
        merged_df.to_csv(os.path.join(directory, "merged.csv"), index=False)
        print(
            f"✅ Merged file saved as '{os.path.join(directory, 'merged.csv')}' with shape {merged_df.shape}"
        )
    else:
        print("❌ No datasets remained after trimming to common range.")
