"""
agent.py — PPO-based RL agent for 5G bandwidth allocation.

Wraps Stable-Baselines3 PPO for training and inference.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from config import (
    PPO_TOTAL_TIMESTEPS,
    PPO_QUICK_TIMESTEPS,
    PPO_LEARNING_RATE,
    PPO_N_STEPS,
    PPO_BATCH_SIZE,
    PPO_N_EPOCHS,
    PPO_MODEL_PATH,
)
from rl_agent.network_env import NetworkSliceEnv


# ─────────────────────────────────────────────
# Custom Logging Callback
# ─────────────────────────────────────────────


class MetricsCallback(BaseCallback):
    """
    Logs per-step metrics (reward, SLA violations, utilization) to a
    shared list so the dashboard can read them live.
    """

    def __init__(self, metrics_store: list, verbose: int = 0):
        super().__init__(verbose)
        self.metrics_store = metrics_store
        self._ep_rewards: list[float] = []

    def _on_step(self) -> bool:
        # Collect infos from vectorised env
        for info in self.locals.get("infos", []):
            sla_ok = info.get("sla_ok", [True, True, True])
            n_viol = sum(1 for ok in sla_ok if not ok)
            alloc = info.get("allocation", {})
            util = info.get("utilization", {})
            reward_val = self.locals.get("rewards", [0.0])
            r = (
                float(reward_val[0])
                if hasattr(reward_val, "__len__")
                else float(reward_val)
            )

            self.metrics_store.append(
                {
                    "step": self.num_timesteps,
                    "reward": r,
                    "sla_violations": n_viol,
                    "alloc_embb": alloc.get("eMBB", 0.33),
                    "alloc_urllc": alloc.get("URLLC", 0.33),
                    "alloc_mmtc": alloc.get("mMTC", 0.34),
                    "util_embb": util.get("eMBB", 0.0),
                    "util_urllc": util.get("URLLC", 0.0),
                    "util_mmtc": util.get("mMTC", 0.0),
                }
            )
        return True


# ─────────────────────────────────────────────
# Agent wrapper
# ─────────────────────────────────────────────


class PPOAgent:
    """
    Wraps Stable-Baselines3 PPO for the 5G slice allocation task.

    Args:
        sim_records     : Simulation records to build the environment
        predictor       : Optional TrafficPredictor (None → no forecast in obs)
        metrics_store   : Shared list for live metric collection
        quick           : If True, train for PPO_QUICK_TIMESTEPS only
    """

    def __init__(
        self,
        sim_records: list[dict],
        predictor=None,
        metrics_store: list | None = None,
        quick: bool = False,
        use_ids_obs: bool = False,
    ):
        self.sim_records = sim_records
        self.predictor = predictor
        self.metrics_store = metrics_store if metrics_store is not None else []
        self.quick = quick
        self.use_ids_obs = use_ids_obs
        self.model: PPO | None = None
        self._env: DummyVecEnv | None = None

    def _make_env(self) -> DummyVecEnv:
        def _init():
            env = NetworkSliceEnv(
                sim_records=self.sim_records,
                predictor=self.predictor,
                use_ids_obs=self.use_ids_obs,
            )
            return Monitor(env)

        return DummyVecEnv([_init])

    def train(self, total_timesteps: int | None = None) -> PPO:
        """Train the PPO agent and save the model."""
        if total_timesteps is None:
            total_timesteps = PPO_QUICK_TIMESTEPS if self.quick else PPO_TOTAL_TIMESTEPS

        print(f"\n[PPO] Training PPO for {total_timesteps:,} timesteps ...")
        vec_env = self._make_env()
        self._env = vec_env

        self.model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=PPO_LEARNING_RATE,
            n_steps=PPO_N_STEPS,
            batch_size=PPO_BATCH_SIZE,
            n_epochs=PPO_N_EPOCHS,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log="logs/tensorboard/",
        )

        callbacks = [MetricsCallback(self.metrics_store)]

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )

        os.makedirs(os.path.dirname(PPO_MODEL_PATH) or ".", exist_ok=True)
        self.model.save(PPO_MODEL_PATH)
        print(f"\n✓ PPO model saved → {PPO_MODEL_PATH}.zip")
        return self.model

    def load(self, path: str = PPO_MODEL_PATH) -> PPO:
        """Load a previously saved PPO model."""
        vec_env = self._make_env()
        self._env = vec_env
        self.model = PPO.load(path, env=vec_env)
        print(f"OK Loaded PPO model from {path}")
        return self.model

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """
        Given an observation, return softmax-normalised allocation fractions.

        Returns:
            np.ndarray of shape (3,) — [α_eMBB, α_URLLC, α_mMTC] summing to 1
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call train() or load() first.")
        action, _ = self.model.predict(obs, deterministic=True)
        action = np.array(action, dtype=np.float64).flatten()[:3]
        # Apply softmax
        action = np.exp(action - action.max())
        return (action / action.sum()).astype(np.float32)

    @staticmethod
    def default_allocation() -> np.ndarray:
        """Fallback uniform allocation when model is not available."""
        return np.array([0.50, 0.30, 0.20], dtype=np.float32)


# ─────────────────────────────────────────────
# SAC Agent wrapper
# ─────────────────────────────────────────────


class SACAgent:
    """
    Wraps Stable-Baselines3 SAC for the 5G slice allocation task.
    """

    def __init__(
        self,
        sim_records: list[dict],
        predictor=None,
        metrics_store: list | None = None,
        quick: bool = False,
        use_ids_obs: bool = False,
    ):
        self.sim_records = sim_records
        self.predictor = predictor
        self.metrics_store = metrics_store if metrics_store is not None else []
        self.quick = quick
        self.use_ids_obs = use_ids_obs
        self.model: SAC | None = None
        self._env: DummyVecEnv | None = None

    def _make_env(self) -> DummyVecEnv:
        def _init():
            env = NetworkSliceEnv(
                sim_records=self.sim_records,
                predictor=self.predictor,
                use_ids_obs=self.use_ids_obs,
            )
            return Monitor(env)

        return DummyVecEnv([_init])

    def train(self, total_timesteps: int | None = None) -> SAC:
        if total_timesteps is None:
            total_timesteps = PPO_QUICK_TIMESTEPS if self.quick else PPO_TOTAL_TIMESTEPS

        print(f"\n[SAC] Training SAC for {total_timesteps:,} timesteps ...")
        vec_env = self._make_env()
        self._env = vec_env

        self.model = SAC(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            buffer_size=100000,
            batch_size=256,
            ent_coef="auto",
            gamma=0.99,
            tau=0.005,
            verbose=1,
            tensorboard_log="logs/tensorboard_sac/",
        )

        callbacks = [MetricsCallback(self.metrics_store)]

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )

        model_path = PPO_MODEL_PATH.replace("ppo", "sac")
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        self.model.save(model_path)
        print(f"\n✓ SAC model saved → {model_path}.zip")
        return self.model

    def load(self, path: str = None) -> SAC:
        if path is None:
            path = PPO_MODEL_PATH.replace("ppo", "sac")
        vec_env = self._make_env()
        self._env = vec_env
        self.model = SAC.load(path, env=vec_env)
        print(f"OK Loaded SAC model from {path}")
        return self.model

    def predict(self, obs: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call train() or load() first.")
        action, _ = self.model.predict(obs, deterministic=True)
        action = np.array(action, dtype=np.float64).flatten()[:3]
        action = np.exp(action - action.max())
        return (action / action.sum()).astype(np.float32)

    @staticmethod
    def default_allocation() -> np.ndarray:
        return np.array([0.50, 0.30, 0.20], dtype=np.float32)
