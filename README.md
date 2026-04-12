# 5G Intelligent Network Resource Management System

A dynamic, fully autonomous reinforcement learning testbed designed to intelligently allocate 5G network slice bandwidth across differing Service Level Agreement (SLA) profiles (eMBB, URLLC, mMTC) while simultaneously fending off simulated and heuristically detected network attacks.

## Core Features
1. **Predictive Analytics (Transformer Focus):** Utilizes an advanced NLP-inspired Self-Attention Transformer Encoder to forecast multi-step resource demands efficiently using chunked mean-pooling.
2. **Dual Reinforcement Learning Pipelines:** Out-of-the-box support for training both Proximal Policy Optimization (**PPO**) and Soft Actor-Critic (**SAC**) algorithms. 
3. **Attack-Aware Observation Signals:** Intrusion detection systems (IDS) logic dynamically weaves sequence-based threat labels directly into the RL observation vectors, effectively penalizing poor URLLC throughput handling during Denial of Service (DoS) sweeps.
4. **Live Flask Dashboard:** Server-Sent Events (SSE) stream forecasting, slice allocation decisions, threat detection states, and cumulative Q-reward mapping to a beautiful frontend dashboard at 2-3 frames per second.
5. **SimPy vs Labeled Datasets:** Seamlessly switch between training via unbounded synthetically generated Markov trajectories (`TrafficSimulator`) or static, real-world derived CSV flows (`final_dataset.csv`).

## Getting Started

### Prerequisites
Make sure to have Python 3.9+ and pip installed. We highly recommend utilizing a virtual environment for installation. 
```bash
pip install -r requirements.txt
```

### Modes & Usage 

The `run.py` script serves as the centralized commander for the whole application. 
You can chain pipeline states sequentially or run them individually. 

```bash
# 1. Quick full-pipeline dataset validation using SAC Agent 
python3 run.py --all --quick --use-dataset --use-ids-obs --agent sac

# 2. Run ablation comparisons between Static/Proportional and RL iterations
python3 run.py --ablation --use-dataset

# 3. Only run live inference on the SSE dashboard without training 
python3 run.py --serve --use-dataset --use-ids-obs
```

**Common Flags:**
- `--pretrain` / `--train-rl` / `--serve`: Invokes atomic sub-stages.
- `--use-dataset`: Read from `dataset/final_dataset.csv` instead of random walk generation.
- `--use-ids-obs`: Expands the observation state to conditionally include attack labels.
- `--quick`: Caps training iterations dramatically for debugging/verification.
- `--agent {ppo, sac}`: Assign deterministic stable-baselines3 algorithms.
