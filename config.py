"""
config.py — Global constants and hyperparameters for the 5G Network Resource Management System.
"""

# ─────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────
NUM_UES = 1000                  # Total user equipments
SIM_DURATION = 500              # Timesteps per simulation run
SLICE_RATIOS = {                # Fraction of UEs per slice
    "eMBB":  0.40,
    "URLLC": 0.30,
    "mMTC":  0.30,
}

# Traffic profiles (Mbps or packet units)
TRAFFIC_PROFILES = {
    "eMBB":  {"mean": 50.0, "burst_scale": 3.0},   # exponential bursty
    "URLLC": {"mean": 2.0,  "interval": 0.05},      # frequent tiny packets
    "mMTC":  {"mean": 0.5,  "interval": 1.0},       # periodic low-rate
}

# ─────────────────────────────────────────────
# Network / Bandwidth
# ─────────────────────────────────────────────
TOTAL_BANDWIDTH_MHZ = 100.0     # Total simulated bandwidth (MHz)
INITIAL_ALLOC = {               # Default initial allocation fractions
    "eMBB":  0.50,
    "URLLC": 0.30,
    "mMTC":  0.20,
}

# SLA thresholds
SLA_THROUGHPUT_EMBB   = 30.0    # Mbps minimum for eMBB
SLA_LATENCY_URLLC_MS  = 1.0     # Max latency in ms for URLLC
SLA_PDR_MMTC          = 0.95    # Minimum packet delivery ratio for mMTC

# ─────────────────────────────────────────────
# Transformer Predictor
# ─────────────────────────────────────────────
WINDOW_T = 20                   # Lookback window (timesteps)
HORIZON_H = 10                  # Forecast horizon
N_FEATURES = 3                  # [eMBB, URLLC, mMTC]
D_MODEL = 64                    # Transformer model dimension
N_HEADS = 4                     # Multi-head attention heads
N_LAYERS = 2                    # Encoder layers
DROPOUT = 0.1
TRANSFORMER_LR = 1e-3
TRANSFORMER_EPOCHS = 60
TRANSFORMER_BATCH = 64
PRETRAIN_STEPS = 5000           # Simulation steps to generate training data

MODEL_WEIGHTS_PATH = "predictor/model_weights.pt"
SCALER_PATH        = "predictor/scaler.pkl"

# ─────────────────────────────────────────────
# RL Agent (PPO)
# ─────────────────────────────────────────────
PPO_TOTAL_TIMESTEPS  = 500_000
PPO_QUICK_TIMESTEPS  = 50_000   # --quick flag
PPO_LEARNING_RATE    = 3e-4
PPO_N_STEPS          = 2048
PPO_BATCH_SIZE       = 64
PPO_N_EPOCHS         = 10
RL_SLA_PENALTY_LAMBDA = 0.5     # Weight of SLA violations in reward
URLLC_VIOLATION_PENALTY = 5.0  # Extra penalty multiplier for URLLC violations

PPO_MODEL_PATH = "rl_agent/ppo_5g_model"

# ─────────────────────────────────────────────
# Dashboard / Flask
# ─────────────────────────────────────────────
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
SSE_INTERVAL = 0.5              # Seconds between SSE pushes
REWARD_WINDOW = 50              # Rolling window for reward smoothing

# ─────────────────────────────────────────────
# Ablation
# ─────────────────────────────────────────────
ABLATION_STEPS = 300            # Steps per ablation run
ABLATION_RESULTS_PATH = "ablation/ablation_results.csv"
