#!/usr/bin/env python3
"""
run.py — Master entry point for the Intelligent 5G Network Resource Management System.

Modes:
  python run.py --pretrain              Pre-train Transformer on simulated data
  python run.py --train-rl              Train PPO RL agent (uses trained transformer)
  python run.py --train-rl --quick      Quick training (50k steps, ~2 min)
  python run.py --serve                 Start live dashboard + pipeline loop
  python run.py --serve --quick         Quick pretrain+train then serve
  python run.py --all                   Full pipeline: pretrain → train RL → serve
  python run.py --all --quick           All phases in quick mode

Dashboard opens automatically at http://localhost:5000
"""

import argparse
import os
import sys
import time
import webbrowser
import threading
import numpy as np

# ── Project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    NUM_UES,
    SIM_DURATION,
    PRETRAIN_STEPS,
    WINDOW_T,
    HORIZON_H,
    D_MODEL,
    MODEL_WEIGHTS_PATH,
    SCALER_PATH,
    PPO_MODEL_PATH,
    FLASK_PORT,
    REWARD_WINDOW,
)


# ─────────────────────────────────────────────
# Phase 1: Pre-train Transformer
# ─────────────────────────────────────────────


def phase_pretrain(sim_records=None, verbose=True, use_dataset=False):
    """Generate simulation data and train the transformer."""
    from predictor.trainer import train_transformer, fast_simulate

    if sim_records is None:
        if use_dataset:
            from dataset.loader import load_sim_records

            print("\n[LOAD] Loading sim_records from final_dataset.csv ...")
            sim_records = load_sim_records()
            print(f"  OK {len(sim_records):,} records loaded from dataset")
        else:
            from simulation.traffic_sim import TrafficSimulator

            print(
                f"\n[FAST] Running fast vectorized simulation ({PRETRAIN_STEPS} steps) ..."
            )
            sim_records = fast_simulate(PRETRAIN_STEPS, num_ues=NUM_UES, seed=0)
            print(f"  OK Generated {len(sim_records)} records")

    print(
        f"\n[TRAIN] Training Transformer (window T={WINDOW_T}, horizon H={HORIZON_H}) ..."
    )
    model, scaler, history = train_transformer(sim_records, verbose=verbose)
    print(f"  OK Final val_loss: {history['val_loss'][-1]:.6f}")
    return model, scaler, sim_records


# ─────────────────────────────────────────────
# Phase 2: Train RL Agent
# ─────────────────────────────────────────────


def phase_train_rl(
    sim_records,
    model=None,
    scaler=None,
    quick=False,
    use_dataset=False,
    agent_type="ppo",
    use_ids_obs=False,
):
    """Train RL agent using the 5G environment."""
    from predictor.trainer import TrafficPredictor
    from predictor.transformer_model import load_model
    from predictor.dataset import SliceDataset
    from rl_agent.agent import PPOAgent, SACAgent

    # Load transformer if not provided
    if model is None and os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"\n[LOAD] Loading transformer from {MODEL_WEIGHTS_PATH} ...")
        model = load_model(
            MODEL_WEIGHTS_PATH,
            device="cpu",
            window_t=WINDOW_T,
            horizon_h=HORIZON_H,
        )
        scaler = SliceDataset.load_scaler(SCALER_PATH)

    # If sim_records not supplied, optionally load from CSV
    if sim_records is None:
        if use_dataset:
            from dataset.loader import load_sim_records

            print("\n[LOAD] Loading RL training records from final_dataset.csv ...")
            sim_records = load_sim_records()
            sim_records = load_sim_records()
        else:
            from simulation.traffic_sim import TrafficSimulator

            print(f"\n📡 Generating simulation data for RL training …")
            sim_records = TrafficSimulator(seed=1).run(duration=PRETRAIN_STEPS)

    predictor = TrafficPredictor(model, scaler) if model is not None else None

    AgentClass = SACAgent if agent_type.lower() == "sac" else PPOAgent

    metrics_store = []
    agent = AgentClass(
        sim_records=sim_records,
        predictor=predictor,
        metrics_store=metrics_store,
        quick=quick,
        use_ids_obs=use_ids_obs,
    )
    agent.train()
    return agent, predictor


# ─────────────────────────────────────────────
# Phase 3: Live pipeline loop
# ─────────────────────────────────────────────


def phase_serve(
    agent=None,
    predictor=None,
    sim_records=None,
    quick=False,
    use_dataset=False,
    agent_type="ppo",
    use_ids_obs=False,
):
    """Start Flask dashboard and run the live pipeline loop."""
    import sys

    print("Starting phase_serve...", flush=True)
    from simulation.traffic_sim import StreamingTrafficSimulator
    from environment.fiveg_env import FiveGEnvironment
    from rl_agent.network_env import NetworkSliceEnv
    from rl_agent.agent import PPOAgent, SACAgent
    from predictor.trainer import TrafficPredictor
    from predictor.transformer_model import load_model
    from predictor.dataset import SliceDataset
    from dashboard.app import run_server, push_frame

    print("All imports done", flush=True)

    # ── Load models if not in memory ─────────
    if predictor is None and os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"\n[LOAD] Loading transformer ...", flush=True)
        model = load_model(
            MODEL_WEIGHTS_PATH,
            device="cpu",
            window_t=WINDOW_T,
            horizon_h=HORIZON_H,
            d_model=D_MODEL,
        )
        print("Model loaded", flush=True)
        scaler = SliceDataset.load_scaler(SCALER_PATH)
        print("Scaler loaded", flush=True)
        predictor = TrafficPredictor(model, scaler)
        print("  OK Transformer loaded", flush=True)

    if agent is None:
        # Need sim_records to build env for agent
        print(f"\n[AGENT] Checking agent ...", flush=True)
        print(f"  sim_records is None: {sim_records is None}", flush=True)
        if sim_records is None:
            from predictor.trainer import fast_simulate

            print(f"\n[SIM] Running fast simulation for RL env ...", flush=True)
            sim_records = fast_simulate(1000, num_ues=NUM_UES, seed=1)

        print(f"\n[AGENT] Building agent class ...")
        AgentClass = SACAgent if agent_type.lower() == "sac" else PPOAgent
        model_path = (
            PPO_MODEL_PATH.replace("ppo", "sac")
            if agent_type.lower() == "sac"
            else PPO_MODEL_PATH
        )
        print(f"\n[AGENT] Model path: {model_path}")
        zip_path = model_path + ".zip"
        if os.path.exists(zip_path):
            print(f"\n[LOAD] Loading {agent_type.upper()} agent from {zip_path} ...")
            agent = AgentClass(
                sim_records=sim_records, predictor=predictor, use_ids_obs=use_ids_obs
            )
            agent.load(model_path)
        else:
            print(
                f"\n[WARN] No trained {agent_type.upper()} model found. Using rule-based allocation."
            )
            agent = None

    # ── Start Flask server ────────────────────
    print(f"\n[SERVER] Starting Flask server ...")
    run_server(port=FLASK_PORT)
    time.sleep(1.5)

    # Auto-open browser (user confirmed yes)
    url = f"http://localhost:{FLASK_PORT}"
    print(f"[OPEN] Opening dashboard at {url}")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    # ── Live pipeline loop ────────────────────
    print("\n[RUN] Pipeline running. Press Ctrl+C to stop.\n")
    if use_dataset and sim_records is not None:
        streaming_sim = sim_records  # Use loaded list as iterator target
        sim_idx = 0
    else:
        streaming_sim = StreamingTrafficSimulator(num_ues=NUM_UES, seed=99)
        sim_idx = None
    fiveg = FiveGEnvironment()

    reward_history: list[float] = []
    step = 0

    # Build a NetworkSliceEnv observation builder
    obs_env = NetworkSliceEnv(
        sim_records=sim_records
        if sim_records
        else [
            {"eMBB": 0, "URLLC": 0, "mMTC": 0, "active_users": [400, 300, 300], "t": 0}
        ],
        predictor=predictor,
        episode_len=99999,
        use_ids_obs=use_ids_obs,
    )
    obs, _ = obs_env.reset(seed=0)

    try:
        while True:
            # Step 1: Get demand from stream or dataset
            if sim_idx is not None:
                record = streaming_sim[sim_idx % len(streaming_sim)]
                sim_idx += 1
            else:
                record = streaming_sim.step()

            demand = {
                "eMBB": record["eMBB"],
                "URLLC": record["URLLC"],
                "mMTC": record["mMTC"],
            }

            # Step 2: Update predictor buffer
            if predictor is not None:
                predictor.update(demand)

            # Step 3: Get forecast
            forecast = None
            if predictor is not None:
                fc_raw = predictor.predict()
                if fc_raw is not None:
                    forecast = fc_raw.tolist()  # (H, 3)

            # Step 4: RL agent selects action
            if agent is not None and agent.model is not None:
                alloc_fracs = agent.predict(obs)
            else:
                # Proportional allocation based on demand
                d = np.array(
                    [demand["eMBB"], demand["URLLC"], demand["mMTC"]], dtype=np.float64
                )
                d_sum = d.sum() + 1e-9
                alloc_fracs = (d / d_sum).astype(np.float32)
                # Floor: URLLC always gets at least 20%
                alloc_fracs[1] = max(alloc_fracs[1], 0.20)
                alloc_fracs /= alloc_fracs.sum()

            # Step 5: Apply allocation in 5G env
            result = fiveg.allocate(demand, alloc_fracs)
            qos = result["qos"]
            sla_ok = result["sla_ok"]

            # Step 6: Compute reward
            qos_score = fiveg.qos_score()
            n_violations = sum(1 for ok in sla_ok if not ok)
            from config import RL_SLA_PENALTY_LAMBDA, URLLC_VIOLATION_PENALTY

            urllc_pen = 0.0 if sla_ok[1] else URLLC_VIOLATION_PENALTY
            reward = (
                qos_score
                - RL_SLA_PENALTY_LAMBDA * n_violations
                - RL_SLA_PENALTY_LAMBDA * urllc_pen
            )
            reward_history.append(reward)

            # Step 7: Advance obs in obs_env
            obs, _, terminated, _, _ = obs_env.step(alloc_fracs)
            if terminated:
                obs, _ = obs_env.reset()

            # Heuristic anomaly score for Attack panel if dataset labels aren't present
            is_attack = "Benign"
            att_type = "Benign"
            if "label" in record:
                is_attack = record["label"]
                att_type = record.get("attack_type", "Unknown")
            elif (
                demand["URLLC"] > 20
                and demand["URLLC"] / max(demand["eMBB"], 1.0) > 0.5
            ):
                is_attack = "Attack"
                att_type = "DoS (Heuristic)"

            # Step 8: Build and push SSE frame
            def to_float32(v):
                if hasattr(v, "item"):
                    return float(v.item())
                return float(v) if v is not None else None

            qos_list = (
                [
                    float(qos.get("eMBB_throughput_Mbps", 0)),
                    float(qos.get("URLLC_latency_ms", 0)),
                    float(qos.get("mMTC_PDR", 0)),
                ]
                if qos
                else None
            )

            frame = {
                "t": step,
                "demand": {k: to_float32(v) for k, v in demand.items()},
                "alloc": {
                    "eMBB": to_float32(alloc_fracs[0]),
                    "URLLC": to_float32(alloc_fracs[1]),
                    "mMTC": to_float32(alloc_fracs[2]),
                },
                "forecast": [[float(x) for x in row] for row in forecast]
                if forecast
                else None,
                "qos": qos_list,
                "sla_ok": [bool(x) for x in sla_ok],
                "sla_violations": int(n_violations),
                "reward": to_float32(reward),
                "reward_avg": to_float32(np.mean(reward_history[-REWARD_WINDOW:])),
                "active_users": list(record["active_users"]),
                "label": is_attack,
                "attack_type": att_type,
            }
            push_frame(frame)

            step += 1
            if step % 50 == 0:
                avg_r = np.mean(reward_history[-REWARD_WINDOW:])
                viol_rate = sum(1 for ok in sla_ok if not ok)
                print(
                    f"  Step {step:05d} | reward={reward:.4f} | avg={avg_r:.4f} | violations={viol_rate}"
                )

            time.sleep(0.4)  # ~2.5 updates/sec

    except KeyboardInterrupt:
        print("\n\n[STOP] Pipeline stopped by user.")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="5G Intelligent Network Resource Management System"
    )
    parser.add_argument("--pretrain", action="store_true", help="Pre-train transformer")
    parser.add_argument("--train-rl", action="store_true", help="Train PPO RL agent")
    parser.add_argument("--serve", action="store_true", help="Run live dashboard")
    parser.add_argument("--all", action="store_true", help="Run all phases in sequence")
    parser.add_argument(
        "--quick", action="store_true", help="Quick training (50k PPO steps)"
    )
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument(
        "--use-dataset",
        action="store_true",
        help="Load training data from dataset/final_dataset.csv instead of SimPy",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="ppo",
        choices=["ppo", "sac"],
        help="Choose RL agent: ppo or sac",
    )
    parser.add_argument(
        "--use-ids-obs",
        action="store_true",
        help="Include attack detection label in observation space",
    )
    args = parser.parse_args()

    model = None
    scaler = None
    sim_records = None
    agent = None
    predictor = None

    if args.ablation:
        from ablation.ablation_study import run_ablation

        run_ablation(quick=args.quick)
        return

    use_dataset = getattr(args, "use_dataset", False)

    if args.all or args.pretrain:
        model, scaler, sim_records = phase_pretrain(
            verbose=True, use_dataset=use_dataset
        )

    if args.all or getattr(args, "train_rl"):
        agent, predictor = phase_train_rl(
            sim_records,
            model,
            scaler,
            quick=args.quick,
            use_dataset=use_dataset,
            agent_type=args.agent,
            use_ids_obs=args.use_ids_obs,
        )

    if args.all or args.serve:
        phase_serve(
            agent=agent,
            predictor=predictor,
            sim_records=sim_records,
            quick=args.quick,
            use_dataset=use_dataset,
            agent_type=args.agent,
            use_ids_obs=args.use_ids_obs,
        )
        return

    if not any(
        [args.pretrain, getattr(args, "train_rl"), args.serve, args.all, args.ablation]
    ):
        parser.print_help()
        print("\n[HINT] Quick start: python run.py --all --quick")


if __name__ == "__main__":
    main()
