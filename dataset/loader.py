"""
loader.py — Load final_dataset.csv into the sim_records format
expected by all project modules.

The sim_records list-of-dicts format is:
    [{"eMBB": float, "URLLC": float, "mMTC": float,
      "active_users": list[int], "t": int,
      "label": str, "attack_type": str, "slice_type": str}, ...]

This format is consumed by:
  - predictor/dataset.py  → build_dataset_from_sim()
  - rl_agent/network_env.py → NetworkSliceEnv(sim_records=...)
  - ablation/ablation_study.py → evaluate_agent()
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Default path ──────────────────────────────────────────────────────────────
_DEFAULT_CSV = Path(__file__).resolve().parent / "final_dataset.csv"


def load_sim_records(
    csv_path: Optional[str] = None,
    shuffle: bool = True,
    seed: int = 42,
    max_rows: Optional[int] = None,
    attacks_only: bool = False,
    benign_only: bool = False,
) -> list[dict]:
    """
    Load final_dataset.csv and return records in the sim_records format.

    Args:
        csv_path    : Path to final_dataset.csv. Defaults to dataset/final_dataset.csv.
        shuffle     : Shuffle rows before returning (preserves i.i.d. for RL).
        seed        : Random seed for shuffle.
        max_rows    : Truncate to this many rows (None = all).
        attacks_only: Return only rows with Label == "Attack".
        benign_only : Return only rows with Label == "Benign".

    Returns:
        List of dicts with keys:
            eMBB, URLLC, mMTC      — float, Mbps demand
            active_users           — [int, int, int] UE counts per slice
            t                      — int, timestep index
            label                  — "Benign" or "Attack"
            attack_type            — e.g. "DoS", "Benign", "Fuzzing" …
            slice_type             — "eMBB" | "URLLC" | "mMTC"
            flow_features          — dict of raw network-flow columns
    """
    path = Path(csv_path) if csv_path else _DEFAULT_CSV

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Run:  python dataset/generate_final_dataset.py"
        )

    df = pd.read_csv(path, nrows=max_rows, low_memory=False)

    # ── Filter ────────────────────────────────────────────────────────────────
    if attacks_only:
        df = df[df["Label"] == "Attack"].reset_index(drop=True)
    elif benign_only:
        df = df[df["Label"] == "Benign"].reset_index(drop=True)

    # ── Shuffle ───────────────────────────────────────────────────────────────
    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    else:
        if "Seq" in df.columns:
            df = df.sort_values(by="Seq").reset_index(drop=True)

    # ── Flow-feature columns (everything except slice demand + labels) ─────────
    _DEMAND_COLS = {"eMBB_demand", "URLLC_demand", "mMTC_demand",
                    "slice_type", "Label", "Attack_Type"}
    flow_cols = [c for c in df.columns if c not in _DEMAND_COLS]

    # ── Build records ─────────────────────────────────────────────────────────
    records: list[dict] = []
    for i, row in df.iterrows():
        records.append({
            # 5G slice demand (used by Transformer & RL env)
            "eMBB":         float(row.get("eMBB_demand", 0.0)),
            "URLLC":        float(row.get("URLLC_demand", 0.0)),
            "mMTC":         float(row.get("mMTC_demand", 0.0)),
            # UE counts (fixed from config defaults)
            "active_users": [400, 300, 300],
            # Timestep index
            "t":            int(i),
            # Label / intrusion detection fields
            "label":        str(row.get("Label", "Benign")),
            "attack_type":  str(row.get("Attack_Type", "Benign")),
            "slice_type":   str(row.get("slice_type", "eMBB")),
            # Full network-flow features (for future IDS module)
            "flow_features": {c: row[c] for c in flow_cols},
        })

    print(
        f"[loader] Loaded {len(records):,} records from {path.name} "
        f"| Benign: {sum(1 for r in records if r['label']=='Benign'):,} "
        f"| Attack: {sum(1 for r in records if r['label']=='Attack'):,}"
    )
    return records


def records_to_demand_array(records: list[dict]) -> "np.ndarray":
    """
    Convert sim_records to a (N, 3) float32 array [eMBB, URLLC, mMTC].
    Convenience helper for the Transformer dataset builder.
    """
    return np.array(
        [[r["eMBB"], r["URLLC"], r["mMTC"]] for r in records],
        dtype=np.float32,
    )
