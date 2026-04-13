"""
trainer.py — Training loop for the Transformer traffic predictor.

Usage:
    python -m predictor.trainer   (standalone pre-training)
    or imported by run.py via train_transformer(sim_records)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Allow running as standalone script
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import (
    WINDOW_T,
    HORIZON_H,
    N_FEATURES,
    D_MODEL,
    N_HEADS,
    N_LAYERS,
    DROPOUT,
    TRANSFORMER_LR,
    TRANSFORMER_EPOCHS,
    TRANSFORMER_BATCH,
    PRETRAIN_STEPS,
    MODEL_WEIGHTS_PATH,
    SCALER_PATH,
)
from predictor.transformer_model import TrafficTransformer
from predictor.dataset import build_dataset_from_sim


# ─────────────────────────────────────────────
# Training function
# ─────────────────────────────────────────────


def train_transformer(
    sim_records: list[dict],
    epochs: int = TRANSFORMER_EPOCHS,
    batch_size: int = TRANSFORMER_BATCH,
    lr: float = TRANSFORMER_LR,
    device: str = "cuda",
    verbose: bool = True,
) -> tuple[TrafficTransformer, object]:
    """
    Train the Transformer on simulation data.

    Args:
        sim_records : List of dicts with 'eMBB','URLLC','mMTC' keys
        epochs      : Training epochs
        batch_size  : Mini-batch size
        lr          : Learning rate
        device      : 'cpu' or 'cuda'
        verbose     : Show progress bar

    Returns:
        (trained_model, scaler)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.cuda.empty_cache()
        _ = torch.zeros(1, device=device)
        print(f"  [GPU] {torch.cuda.get_device_name(0)}")

    # Build dataset
    dataset = build_dataset_from_sim(sim_records, T=WINDOW_T, H=HORIZON_H)
    if len(dataset) < 10:
        raise ValueError(
            f"Dataset too small ({len(dataset)} samples). "
            f"Run simulation for at least {WINDOW_T + HORIZON_H + 10} steps."
        )

    # Train/val split (90/10)
    val_len = max(1, int(0.1 * len(dataset)))
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    # Model
    model = TrafficTransformer(
        n_features=N_FEATURES,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        window_t=WINDOW_T,
        horizon_h=HORIZON_H,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", disable=not verbose)
        for x_batch, y_batch in pbar:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.5f}")

        # ── Validate ──
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch)
                val_losses.append(criterion(pred, y_batch).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step()

        if verbose:
            print(
                f"  Epoch {epoch:03d} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f}"
            )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Persist
    os.makedirs(os.path.dirname(MODEL_WEIGHTS_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
    dataset.save_scaler(SCALER_PATH)
    if verbose:
        print(f"\nOK Model saved -> {MODEL_WEIGHTS_PATH}")
        print(f"OK Scaler saved -> {SCALER_PATH}")

    return model.eval(), dataset.scaler, history


# ─────────────────────────────────────────────
# Inference wrapper
# ─────────────────────────────────────────────


class TrafficPredictor:
    """
    Thin wrapper around a trained TrafficTransformer.
    Handles normalisation ↔ denormalisation automatically.
    """

    def __init__(self, model: TrafficTransformer, scaler, device: str = None):
        self.model = model.eval()
        self.scaler = scaler
        self.device = (
            torch.device(next(model.parameters()).device)
            if device is None
            else torch.device(device)
        )
        self.buffer: list[np.ndarray] = []  # rolling window buffer
        self.T = model.window_t

    def update(self, demand: dict[str, float]):
        """Push a new timestep observation into the rolling buffer."""
        vec = np.array(
            [demand["eMBB"], demand["URLLC"], demand["mMTC"]], dtype=np.float32
        )
        self.buffer.append(vec)
        if len(self.buffer) > self.T:
            self.buffer.pop(0)

    def predict(self) -> np.ndarray | None:
        """
        Return forecast array of shape (H, 3) in original scale,
        or None if buffer not yet full.
        """
        if len(self.buffer) < self.T:
            return None
        window = np.stack(self.buffer, axis=0)  # (T, 3)
        window_norm = self.scaler.transform(window)  # normalised
        x = torch.tensor(window_norm, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred_norm = self.model(x).squeeze(0).cpu().numpy()  # (H, 3)
        # Inverse transform each step
        pred = self.scaler.inverse_transform(pred_norm)  # (H, 3)
        return pred


# ─────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────


def fast_simulate(num_steps: int, num_ues: int = 1000, seed: int = 42) -> list[dict]:
    """Fast vectorized simulation using numpy (no SimPy)."""
    rng = np.random.default_rng(seed)
    n_embb = int(num_ues * 0.4)
    n_urllc = int(num_ues * 0.3)
    n_mmtc = num_ues - n_embb - n_urllc

    t = np.arange(num_steps)
    embb = rng.exponential(50, num_steps) * rng.choice(
        [1.0, 3.0], num_steps, p=[0.8, 0.2]
    )
    embb += n_embb * 2

    urllc = 2.0 + rng.uniform(-0.4, 0.4, num_steps)
    urllc *= n_urllc * 10

    mmtc = 0.5 + rng.uniform(-0.05, 0.05, num_steps)
    mmtc *= n_mmtc

    return [
        {
            "eMBB": round(e, 4),
            "URLLC": round(u, 4),
            "mMTC": round(m, 4),
            "active_users": [n_embb, n_urllc, n_mmtc],
            "t": ti,
        }
        for ti, e, u, m in zip(t, embb, urllc, mmtc)
    ]


if __name__ == "__main__":
    print("Pre-training Transformer on fresh simulation data ...")
    print(f"  Using fast vectorized simulation ({PRETRAIN_STEPS} steps) ...")
    records = fast_simulate(PRETRAIN_STEPS, seed=0)
    train_transformer(records, verbose=True)
