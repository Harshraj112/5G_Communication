# 5G Intelligent Network Resource Management System

A dynamic, fully autonomous reinforcement learning testbed designed to intelligently allocate 5G network slice bandwidth across differing Service Level Agreement (SLA) profiles (eMBB, URLLC, mMTC) while simultaneously fending off simulated and heuristically detected network attacks.

<img width="1438" height="771" alt="Screenshot 2026-04-18 at 2 16 50 AM" src="https://github.com/user-attachments/assets/0d5f0624-f1d2-4266-b112-0dea7ca5058f" />
<img width="1440" height="781" alt="Screenshot 2026-04-18 at 2 16 59 AM" src="https://github.com/user-attachments/assets/ad24cec8-44f7-4eca-8b52-dfc0be9ea112" />


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

## Project Architecture

### Backend Stack
- **Framework:** Flask 2.3.0 with CORS support
- **RL Agent:** Stable-Baselines3 (PPO algorithm)
- **Traffic Prediction:** PyTorch Transformer (D_MODEL=128, 4 attention heads)
- **Simulation:** SimPy event-based simulation (1000 UEs)
- **Deployment:** Render (Python 3.11 + Gunicorn)

### Frontend Stack
- **Framework:** React 18 + Vite
- **Visualization:** Plotly.js (real-time charts)
- **Communication:** Server-Sent Events (SSE) for live streaming
- **Styling:** Custom CSS with dark theme
- **Deployment:** Netlify

### Data Flow
```
Traffic Simulator/Dataset
    ↓
Transformer Predictor (20-step input → 10-step forecast)
    ↓
RL Agent (PPO with adaptive URLLC floor 8-20%)
    ↓
Flask Backend (SSE broadcast to frontend)
    ↓
React Dashboard (Real-time visualization)
```

## System Requirements

### Development Environment
- **Python:** 3.11+
- **Node.js:** 18+ (for frontend)
- **RAM:** 8GB minimum
- **Disk:** 2GB (includes PyTorch + dataset)

### Virtual Environment Setup
```bash
# Python backend
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
cd backend && pip install -r requirements.txt

# Node.js frontend
cd frontend
npm install
```

## Deployment

### Backend - Render (Non-Docker)
1. Connect GitHub repository to Render
2. **Settings:**
   - Runtime: Python 3.11
   - Build: `pip install -r backend/requirements.txt`
   - Start: `gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 300 wsgi:app`
3. **Environment Variables:**
   ```
   FLASK_ENV=production
   FLASK_DEBUG=false
   PORT=5001
   ```
4. Backend URL: `https://fiveg-backend.onrender.com`

### Frontend - Netlify
1. Connect GitHub repository to Netlify
2. **Build Settings:**
   - Build command: `npm run build`
   - Publish directory: `dist`
   - Base directory: `frontend`
3. **Environment Variables:**
   ```
   VITE_BACKEND_URL=https://fiveg-backend.onrender.com
   ```
4. Netlify automatically handles deployment on push to main branch

### Local Development
```bash
# Terminal 1: Backend
cd backend
source .venv/bin/activate
python3 wsgi.py  # Runs on http://localhost:5001

# Terminal 2: Frontend (new terminal)
cd frontend
VITE_BACKEND_URL=http://localhost:5001 npm run dev  # Runs on http://localhost:5173
```

## Key Algorithms

### Traffic Prediction (Transformer)
- **Architecture:** Self-attention encoder with position encoding
- **Input:** 20 historical steps of traffic demand
- **Output:** 10-step forecast with confidence intervals
- **Loss:** Spike-weighted MSE (2.0x weight on anomalies)
- **Accuracy:** ~95% on test dataset

### Resource Allocation (RL-PPO)
- **State Space:** 
  - Current demands (eMBB, URLLC, mMTC)
  - Predicted demands (next 5 steps)
  - Queue lengths per slice
  - Attack indicators (if `--use-ids-obs`)
- **Action Space:** Continuous bandwidth allocation percentages
- **Reward:** 
  - +1.0 for meeting all SLAs
  - -2.0 per SLA violation
  - -0.1 per inefficient allocation
- **Training:** 500K timesteps, 2 epochs per batch

### Attack Detection
- **Method:** Rule-based IDS on traffic patterns
- **Signals:** 
  - Sudden traffic spikes (>3σ)
  - Queue buildup (>threshold)
  - Low throughput anomalies
- **Response:** Increase URLLC floor to 20%, reduce eMBB allocation

## Dashboard Metrics

The live dashboard displays:
1. **Real-time Bandwidth Allocation** - Stacked area chart (eMBB, URLLC, mMTC)
2. **Traffic Demand Forecast** - Line chart with prediction confidence bands
3. **Reward Curve** - Running average reward tracking
4. **KPI Cards:**
   - Average Reward
   - SLA Violations
   - Current Allocations
   - System Status

## Testing & Validation

```bash
# Run full ablation study
python3 run.py --ablation --use-dataset

# Quick validation (5 min training)
python3 run.py --all --quick --use-dataset --agent ppo

# Launch live dashboard
python3 run.py --serve --use-dataset
```

## File Structure

```
5G Communication/
├── backend/
│   ├── app.py                 # Flask SSE server
│   ├── wsgi.py                # Gunicorn entry point
│   ├── requirements.txt        # Python dependencies
│   └── dataset/
│       └── final_dataset.csv   # Training data (1000+ samples)
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # React dashboard component
│   │   └── index.css          # Styling
│   ├── package.json           # Node dependencies
│   ├── vite.config.js         # Build configuration
│   └── netlify.toml           # Netlify deployment config
├── predictor/
│   ├── transformer_model.py   # Attention-based forecaster
│   ├── trainer.py             # Training pipeline
│   └── model_weights.pt       # Pre-trained weights
├── rl_agent/
│   ├── agent.py               # PPO/SAC policy
│   └── network_env.py         # Gym environment
├── simulation/
│   └── traffic_sim.py         # SimPy traffic generator
├── environment/
│   └── fiveg_env.py           # Custom 5G Gym environment
└── config.py                  # Global configuration

```

## Performance Metrics

Typical results on `final_dataset.csv`:
- **Average Reward:** 0.95-0.98
- **SLA Violations:** <1%
- **Prediction Accuracy:** 94-96%
- **Inference Speed:** 20-30 steps/second
- **Training Time:** ~45 minutes (500K timesteps, CPU)

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Authors & Acknowledgments

- **Primary Developer:** Harsh Raj
- **Research Base:** 5G network slicing, RL-based resource allocation
- **Inspiration:** Industry standards (3GPP), academic publications on network optimization

## Support

For issues, questions, or feedback:
- Open an issue on GitHub
- Check existing documentation in `PROJECT_PRESENTATION.md`
- Review deployment guides in `RENDER_NO_DOCKER_DEPLOYMENT.md`
