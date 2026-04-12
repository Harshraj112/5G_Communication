"""
ablation_study.py — Compare PPO with Transformer vs PPO without Transformer.

Metrics compared:
  - Average reward
  - SLA violation rate
  - Per-slice QoS averages

Outputs:
  - ablation/ablation_results.csv
  - Printed summary table
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import (
    ABLATION_STEPS, ABLATION_RESULTS_PATH,
    PPO_QUICK_TIMESTEPS, PRETRAIN_STEPS, WINDOW_T,
)
from simulation.traffic_sim import TrafficSimulator
from environment.fiveg_env import FiveGEnvironment
from rl_agent.network_env import NetworkSliceEnv


# ─────────────────────────────────────────────
# Evaluation runner
# ─────────────────────────────────────────────

def evaluate_agent(model, sim_records: list[dict], predictor=None,
                   n_steps: int = ABLATION_STEPS) -> dict:
    """
    Roll out a trained agent for n_steps and collect metrics.

    Args:
        model       : Trained PPO model (or None → random baseline)
        sim_records : Simulation records for environment
        predictor   : TrafficPredictor (None → no forecast)
        n_steps     : Number of evaluation steps

    Returns:
        Dict with aggregated metrics
    """
    env = NetworkSliceEnv(sim_records=sim_records, predictor=predictor, episode_len=n_steps)
    obs, _ = env.reset(seed=99)

    rewards        = []
    sla_violations = []
    qos_embb       = []
    qos_urllc      = []
    qos_mmtc       = []

    for _ in range(n_steps):
        if isinstance(model, str):
            if model == "random":
                action = env.action_space.sample()
            elif model == "proportional":
                idx = min(env._record_ptr + env._step_idx, len(env.sim_records) - 1)
                record = env.sim_records[idx]
                d = np.array([record.get("eMBB",0), record.get("URLLC",0), record.get("mMTC",0)], dtype=np.float64)
                d_sum = d.sum() + 1e-9
                alloc = d / d_sum
                alloc[1] = max(alloc[1], 0.20)
                alloc /= alloc.sum()
                action = np.log(alloc + 1e-9)
            elif model == "static":
                action = np.log(np.array([0.5, 0.3, 0.2]) + 1e-9)
            else:
                action = env.action_space.sample()
        elif model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        sla_ok = info.get("sla_ok", [True, True, True])
        sla_violations.append(sum(1 for ok in sla_ok if not ok))

        qos = info.get("qos", {})
        qos_embb.append(qos.get("eMBB_throughput_Mbps", 0.0))
        qos_urllc.append(qos.get("URLLC_latency_ms", 0.0))
        qos_mmtc.append(qos.get("mMTC_PDR", 0.0))

        if terminated:
            break

    return {
        "avg_reward":           np.mean(rewards),
        "sla_violation_rate":   np.mean(sla_violations),
        "avg_embb_throughput":  np.mean(qos_embb),
        "avg_urllc_latency_ms": np.mean(qos_urllc),
        "avg_mmtc_pdr":         np.mean(qos_mmtc),
    }


# ─────────────────────────────────────────────
# Main ablation logic
# ─────────────────────────────────────────────

def run_ablation(quick: bool = True, verbose: bool = True, use_dataset: bool = False) -> pd.DataFrame:
    """
    Run the full ablation study.

    Steps:
      1. Generate simulation data
      2. Train transformer (for WITH-transformer case)
      3. Train PPO WITH transformer forecast in obs
      4. Train PPO WITHOUT transformer forecast in obs
      5. Evaluate both agents
      6. Compare and export results
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from predictor.trainer import train_transformer, TrafficPredictor
    from config import PPO_LEARNING_RATE, PPO_N_STEPS, PPO_BATCH_SIZE, PPO_N_EPOCHS

    ts = PPO_QUICK_TIMESTEPS if quick else 300_000
    sim_steps = PRETRAIN_STEPS if not quick else 2000

    print("=" * 60)
    print("5G Ablation Study: Transformer vs. No-Transformer")
    print("=" * 60)

    # ── Step 1: Simulation data ──────────────
    print("\n[1/5] Generating/Loading data ...")
    if use_dataset:
        from dataset.loader import load_sim_records
        sim_records = load_sim_records(max_rows=max(sim_steps, 5000))
        print(f"  Loaded {len(sim_records)} timesteps from dataset")
    else:
        sim = TrafficSimulator(seed=42)
        sim_records = sim.run(duration=sim_steps)
        print(f"  Generated {len(sim_records)} timesteps")

    # ── Step 2: Train Transformer ────────────
    print("\n[2/5] Pre-training Transformer ...")
    tf_model, scaler, _ = train_transformer(sim_records, verbose=verbose)
    predictor = TrafficPredictor(tf_model, scaler)

    # ── Step 3: Train PPO WITH Transformer ───
    print(f"\n[3/5] Training PPO WITH Transformer ({ts:,} steps) ...")

    def make_env_with(pred):
        def _init():
            e = NetworkSliceEnv(sim_records=sim_records, predictor=pred)
            return Monitor(e)
        return DummyVecEnv([_init])

    def make_env_without():
        def _init():
            e = NetworkSliceEnv(sim_records=sim_records, predictor=None)
            return Monitor(e)
        return DummyVecEnv([_init])

    ppo_with = PPO(
        "MlpPolicy", make_env_with(predictor),
        learning_rate=PPO_LEARNING_RATE, n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE, n_epochs=PPO_N_EPOCHS, verbose=0,
    )
    ppo_with.learn(total_timesteps=ts, progress_bar=verbose)
    print("  ✓ WITH-transformer agent trained")

    # ── Step 4: Train PPO WITHOUT Transformer ─
    print(f"\n[4/5] Training PPO WITHOUT Transformer ({ts:,} steps) ...")
    ppo_without = PPO(
        "MlpPolicy", make_env_without(),
        learning_rate=PPO_LEARNING_RATE, n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE, n_epochs=PPO_N_EPOCHS, verbose=0,
    )
    ppo_without.learn(total_timesteps=ts, progress_bar=verbose)
    print("  ✓ WITHOUT-transformer agent trained")

    # ── Step 5: Evaluate ─────────────────────
    print("\n[5/5] Evaluating agents ...")
    if use_dataset:
        from dataset.loader import load_sim_records
        eval_records = load_sim_records(seed=123)
    else:
        eval_records = TrafficSimulator(seed=123).run(duration=ABLATION_STEPS + WINDOW_T + 10)

    metrics_with    = evaluate_agent(ppo_with,    eval_records, predictor, ABLATION_STEPS)
    metrics_without = evaluate_agent(ppo_without, eval_records, None,      ABLATION_STEPS)
    metrics_random  = evaluate_agent("random",    eval_records, None,      ABLATION_STEPS)
    metrics_prop    = evaluate_agent("proportional", eval_records, None,   ABLATION_STEPS)
    metrics_static  = evaluate_agent("static",    eval_records, None,      ABLATION_STEPS)

    # ── Step 6: Results table ─────────────────
    results = pd.DataFrame({
        "Metric": [
            "Avg Reward",
            "SLA Violation Rate (violations/step)",
            "Avg eMBB Throughput (Mbps)",
            "Avg URLLC Latency (ms)",
            "Avg mMTC PDR",
        ],
        "With Transformer (PPO)": [
            f"{metrics_with['avg_reward']:.4f}",
            f"{metrics_with['sla_violation_rate']:.4f}",
            f"{metrics_with['avg_embb_throughput']:.2f}",
            f"{metrics_with['avg_urllc_latency_ms']:.4f}",
            f"{metrics_with['avg_mmtc_pdr']:.4f}",
        ],
        "Without Transformer (PPO)": [
            f"{metrics_without['avg_reward']:.4f}",
            f"{metrics_without['sla_violation_rate']:.4f}",
            f"{metrics_without['avg_embb_throughput']:.2f}",
            f"{metrics_without['avg_urllc_latency_ms']:.4f}",
            f"{metrics_without['avg_mmtc_pdr']:.4f}",
        ],
        "Proportional Baseline": [
            f"{metrics_prop['avg_reward']:.4f}",
            f"{metrics_prop['sla_violation_rate']:.4f}",
            f"{metrics_prop['avg_embb_throughput']:.2f}",
            f"{metrics_prop['avg_urllc_latency_ms']:.4f}",
            f"{metrics_prop['avg_mmtc_pdr']:.4f}",
        ],
        "Static Baseline": [
            f"{metrics_static['avg_reward']:.4f}",
            f"{metrics_static['sla_violation_rate']:.4f}",
            f"{metrics_static['avg_embb_throughput']:.2f}",
            f"{metrics_static['avg_urllc_latency_ms']:.4f}",
            f"{metrics_static['avg_mmtc_pdr']:.4f}",
        ],
        "Random Baseline": [
            f"{metrics_random['avg_reward']:.4f}",
            f"{metrics_random['sla_violation_rate']:.4f}",
            f"{metrics_random['avg_embb_throughput']:.2f}",
            f"{metrics_random['avg_urllc_latency_ms']:.4f}",
            f"{metrics_random['avg_mmtc_pdr']:.4f}",
        ],
    })

    # Save CSV
    os.makedirs(os.path.dirname(ABLATION_RESULTS_PATH) or ".", exist_ok=True)
    results.to_csv(ABLATION_RESULTS_PATH, index=False)

    print("\n" + "=" * 60)
    print("ABLATION RESULTS")
    print("=" * 60)
    print(results.to_string(index=False))
    print(f"\n✓ Results saved → {ABLATION_RESULTS_PATH}")

    return results


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Run full training (300k steps per agent). Default: quick mode.")
    parser.add_argument("--use-dataset", action="store_true",
                        help="Use real dataset for ablation study.")
    args = parser.parse_args()
    run_ablation(quick=not args.full, use_dataset=args.use_dataset)
