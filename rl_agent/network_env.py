"""
network_env.py — Gymnasium environment for 5G network slice management.

Observation space:
    [current_demand(3) + forecast(H*3) + sla_status(3)] = 3 + H*3 + 3

Action space:
    Box(3,) — raw logits; softmax applied internally to get allocation fractions

Reward:
    QoS_score - λ * SLA_violations  (URLLC violations penalised extra)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import (
    WINDOW_T, HORIZON_H, N_FEATURES,
    RL_SLA_PENALTY_LAMBDA, URLLC_VIOLATION_PENALTY,
    INITIAL_ALLOC,
)
from environment.fiveg_env import FiveGEnvironment


class NetworkSliceEnv(gym.Env):
    """
    Custom Gymnasium environment for intelligent 5G slice resource management.

    The agent learns to allocate bandwidth fractions across three slices
    (eMBB, URLLC, mMTC) to maximise QoS while minimising SLA violations.

    Args:
        sim_records  : Pre-generated simulation records for episode sampling
        predictor    : Optional TrafficPredictor; None → use dummy forecast
        horizon_h    : Forecast horizon H
        episode_len  : Number of steps per training episode
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        sim_records: list[dict],
        predictor=None,
        horizon_h: int = HORIZON_H,
        episode_len: int = 200,
        use_ids_obs: bool = False,
    ):
        super().__init__()

        self.sim_records  = sim_records
        self.predictor    = predictor
        self.horizon_h    = horizon_h
        self.episode_len  = episode_len
        self.use_ids_obs  = use_ids_obs
        self.fiveg        = FiveGEnvironment()

        # Observation dimensionality
        obs_dim = N_FEATURES + horizon_h * N_FEATURES + N_FEATURES  # 3 + H*3 + 3
        if self.use_ids_obs:
            obs_dim += 1

        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action: 3 raw values → softmax → allocation fractions
        self.action_space = spaces.Box(
            low=-3.0, high=3.0, shape=(N_FEATURES,), dtype=np.float32
        )

        # Internal state
        self._step_idx      = 0
        self._record_ptr    = 0
        self._history: list[dict] = []
        self._episode_reward = 0.0

    # ─────────────────────────────────────────
    # Reset
    # ─────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Random start position in sim_records (leave room for episode_len steps)
        max_start = max(0, len(self.sim_records) - self.episode_len - WINDOW_T)
        self._record_ptr = int(self.np_random.integers(0, max(max_start, 1)))
        self._step_idx   = 0
        self._history    = []
        self._episode_reward = 0.0

        # Warm up predictor buffer
        if self.predictor is not None:
            self.predictor.buffer.clear()
            for i in range(min(WINDOW_T, self._record_ptr + 1)):
                rec = self.sim_records[max(0, self._record_ptr - WINDOW_T + i)]
                self.predictor.update(rec)

        obs = self._get_obs()
        return obs, {}

    # ─────────────────────────────────────────
    # Step
    # ─────────────────────────────────────────

    def step(self, action: np.ndarray):
        # Softmax normalisation of raw action
        action = np.array(action, dtype=np.float64)
        action = np.exp(action - action.max())
        action = action / action.sum()

        # Safety floor: URLLC always gets at least 15% bandwidth
        # (prevents degenerate zero-allocation causing latency blow-up)
        action[1] = max(action[1], 0.15)
        action = action / action.sum()

        # Get current demand from simulation records
        idx = min(self._record_ptr + self._step_idx, len(self.sim_records) - 1)
        record = self.sim_records[idx]
        demand = {"eMBB": record["eMBB"], "URLLC": record["URLLC"], "mMTC": record["mMTC"]}

        # Update predictor buffer
        if self.predictor is not None:
            self.predictor.update(demand)

        # Apply allocation in 5G environment
        result = self.fiveg.allocate(demand, action)

        # Compute reward
        qos     = self.fiveg.qos_score()
        sla_ok  = result["sla_ok"]
        n_violations = sum(1 for ok in sla_ok if not ok)

        # URLLC violation penalty is amplified
        urllc_violation = 0.0 if sla_ok[1] else URLLC_VIOLATION_PENALTY

        reward = (
            qos
            - RL_SLA_PENALTY_LAMBDA * n_violations
            - RL_SLA_PENALTY_LAMBDA * urllc_violation
        )
        reward = float(reward)
        self._episode_reward += reward

        # Store history
        self._history.append({
            "t":       idx,
            "demand":  demand,
            "action":  action.tolist(),
            "qos":     result["qos"],
            "sla_ok":  sla_ok,
            "reward":  reward,
            "alloc":   result["allocation_fracs"],
        })

        self._step_idx += 1
        terminated = self._step_idx >= self.episode_len
        obs        = self._get_obs()

        info = {
            "qos":              result["qos"],
            "sla_ok":           sla_ok,
            "allocation":       result["allocation_fracs"],
            "utilization":      result["utilization"],
            "episode_reward":   self._episode_reward,
        }
        return obs, reward, terminated, False, info

    # ─────────────────────────────────────────
    # Observation builder
    # ─────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        idx    = min(self._record_ptr + self._step_idx, len(self.sim_records) - 1)
        record = self.sim_records[idx]
        demand = np.array([record["eMBB"], record["URLLC"], record["mMTC"]], dtype=np.float32)

        # Scale demand to [0,1] using 99th-percentile max from final_dataset.csv
        # eMBB ≈ 500 Mbps max, URLLC ≈ 25 Mbps max, mMTC ≈ 5 Mbps max
        demand_scaled = demand / np.array([500.0, 25.0, 5.0], dtype=np.float32)
        demand_scaled = np.clip(demand_scaled, 0.0, 1.0)

        # Forecast
        if self.predictor is not None and len(self.predictor.buffer) >= WINDOW_T:
            forecast_raw = self.predictor.predict()   # (H, 3)
            if forecast_raw is not None:
                forecast_scaled = forecast_raw / np.array([500.0, 25.0, 5.0])
                forecast_scaled = np.clip(forecast_scaled, 0.0, 1.0).flatten()
            else:
                forecast_scaled = np.zeros(self.horizon_h * N_FEATURES, dtype=np.float32)
        else:
            forecast_scaled = np.zeros(self.horizon_h * N_FEATURES, dtype=np.float32)

        sla_status = np.array(self.fiveg.last_sla_ok, dtype=np.float32)

        pieces = [demand_scaled, forecast_scaled, sla_status]
        
        if self.use_ids_obs:
            is_attack = 1.0 if record.get("label", "Benign") == "Attack" else 0.0
            pieces.append(np.array([is_attack], dtype=np.float32))

        obs = np.concatenate(pieces).astype(np.float32)
        return obs

    # ─────────────────────────────────────────
    # Render (no-op — dashboard handles viz)
    # ─────────────────────────────────────────

    def render(self):
        pass

    def close(self):
        pass
