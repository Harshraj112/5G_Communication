"""
dataset.py — Sliding-window PyTorch Dataset for training the Transformer.

Builds (X, y) pairs from a list of demand snapshots:
  X shape: (T, 3)  — input window
  y shape: (H, 3)  — target forecast
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


class SliceDataset(Dataset):
    """
    Sliding-window dataset built from a list of demand snapshots.

    Args:
        data   : np.ndarray of shape (N, 3) — raw demand values
        T      : Lookback window length
        H      : Forecast horizon
        scaler : Optional pre-fitted MinMaxScaler; fitted to data if None
    """

    def __init__(
        self,
        data: np.ndarray,
        T: int = 20,
        H: int = 10,
        scaler: MinMaxScaler | None = None,
    ):
        self.T = T
        self.H = H

        # Fit or apply scaler
        if scaler is None:
            self.scaler = MinMaxScaler(feature_range=(0.0, 1.0))
            self.scaler.fit(data)
        else:
            self.scaler = scaler

        self.data_norm = self.scaler.transform(data).astype(np.float32)
        self.N = len(self.data_norm)

    def __len__(self) -> int:
        return max(0, self.N - self.T - self.H + 1)

    def __getitem__(self, idx: int):
        x = self.data_norm[idx : idx + self.T]         # (T, 3)
        y = self.data_norm[idx + self.T : idx + self.T + self.H]  # (H, 3)
        return torch.tensor(x), torch.tensor(y)

    def save_scaler(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.scaler, path)

    @staticmethod
    def load_scaler(path: str) -> MinMaxScaler:
        return joblib.load(path)


def build_dataset_from_sim(sim_records: list[dict], T: int, H: int,
                            scaler=None) -> "SliceDataset":
    """
    Convert simulation output records into a SliceDataset.

    Args:
        sim_records : List of dicts with keys 'eMBB', 'URLLC', 'mMTC'
        T           : Window length
        H           : Horizon
        scaler      : Optional pre-fitted scaler
    """
    arr = np.array([[r["eMBB"], r["URLLC"], r["mMTC"]] for r in sim_records],
                   dtype=np.float32)
    return SliceDataset(arr, T=T, H=H, scaler=scaler)
