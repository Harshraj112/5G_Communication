"""
traffic_sim.py — SimPy-based discrete event traffic simulator for 5G slices.

Simulates 1000+ UEs across three slices:
  - eMBB  : high bandwidth, exponentially bursty
  - URLLC : low latency, frequent tiny packets
  - mMTC  : periodic, low-rate IoT devices
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import simpy
import numpy as np
from config import NUM_UES, SLICE_RATIOS, TRAFFIC_PROFILES, SIM_DURATION


# ─────────────────────────────────────────────
# UE process coroutines
# ─────────────────────────────────────────────


def ue_process(
    env: simpy.Environment,
    ue_id: int,
    slice_type: str,
    demand_store: dict,
    rng: np.random.Generator,
):
    """
    Simpy process for a single UE.
    Continuously generates traffic demand and accumulates it into demand_store.

    Args:
        env         : SimPy environment
        ue_id       : Unique UE identifier
        slice_type  : 'eMBB' | 'URLLC' | 'mMTC'
        demand_store: Shared dict {slice_type: [demand_per_second]}
        rng         : NumPy random generator (for reproducibility)
    """
    profile = TRAFFIC_PROFILES[slice_type]

    while True:
        t = int(env.now)

        if slice_type == "eMBB":
            # Bursty: exponential inter-arrival, high bandwidth
            demand = rng.exponential(profile["mean"]) * rng.choice(
                [1.0, profile["burst_scale"]], p=[0.8, 0.2]
            )
            inter_arrival = rng.exponential(0.5)  # avg 0.5 s between sends

        elif slice_type == "URLLC":
            # Frequent tiny packets, low inter-arrival
            demand = profile["mean"] * rng.uniform(0.8, 1.2)
            inter_arrival = profile["interval"]  # 50 ms

        else:  # mMTC
            # Periodic low-rate
            demand = profile["mean"] * rng.uniform(0.9, 1.1)
            inter_arrival = profile["interval"]  # 1 s

        # Thread-safe accumulation
        demand_store[slice_type][t] = demand_store[slice_type].get(t, 0.0) + demand
        yield env.timeout(inter_arrival)


# ─────────────────────────────────────────────
# Traffic Simulator class
# ─────────────────────────────────────────────


class TrafficSimulator:
    """
    Runs SimPy simulation and produces per-timestep demand snapshots.
    """

    def __init__(self, num_ues: int = NUM_UES, seed: int = 42):
        self.num_ues = num_ues
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Assign slices
        n_embb = int(num_ues * SLICE_RATIOS["eMBB"])
        n_urllc = int(num_ues * SLICE_RATIOS["URLLC"])
        n_mmtc = num_ues - n_embb - n_urllc

        self.ue_slices = (
            [("eMBB", i) for i in range(n_embb)]
            + [("URLLC", n_embb + i) for i in range(n_urllc)]
            + [("mMTC", n_embb + n_urllc + i) for i in range(n_mmtc)]
        )

    def run(self, duration: int = SIM_DURATION) -> list[dict]:
        """
        Run the simulation for `duration` timesteps.
        Returns a list of dicts, one per integer timestep.
        """
        env = simpy.Environment()
        demand_store: dict[str, dict[int, float]] = {
            "eMBB": {},
            "URLLC": {},
            "mMTC": {},
        }

        # Launch UE processes
        for slice_type, ue_id in self.ue_slices:
            env.process(ue_process(env, ue_id, slice_type, demand_store, self.rng))

        env.run(until=duration)

        # Build list of per-timestep observations
        results = []
        n_embb = sum(1 for s, _ in self.ue_slices if s == "eMBB")
        n_urllc = sum(1 for s, _ in self.ue_slices if s == "URLLC")
        n_mmtc = sum(1 for s, _ in self.ue_slices if s == "mMTC")

        for t in range(duration):
            embb_d = demand_store["eMBB"].get(t, 0.0)
            urllc_d = demand_store["URLLC"].get(t, 0.0)
            mmtc_d = demand_store["mMTC"].get(t, 0.0)
            results.append(
                {
                    "eMBB": round(embb_d, 4),
                    "URLLC": round(urllc_d, 4),
                    "mMTC": round(mmtc_d, 4),
                    "active_users": [n_embb, n_urllc, n_mmtc],
                    "t": t,
                }
            )

        return results


# ─────────────────────────────────────────────
# Streaming generator for live pipeline
# ─────────────────────────────────────────────


class StreamingTrafficSimulator:
    """
    Incrementally advances the SimPy simulation one step at a time.
    Used by the live pipeline in run.py.
    """

    def __init__(self, num_ues: int = NUM_UES, seed: int = 42):
        self.num_ues = num_ues
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._demand_store: dict[str, dict[int, float]] = {
            "eMBB": {},
            "URLLC": {},
            "mMTC": {},
        }
        self._env = simpy.Environment()
        self._ue_counts = {}
        self._setup()

    def _setup(self):
        n_embb = int(self.num_ues * SLICE_RATIOS["eMBB"])
        n_urllc = int(self.num_ues * SLICE_RATIOS["URLLC"])
        n_mmtc = self.num_ues - n_embb - n_urllc
        self._ue_counts = {"eMBB": n_embb, "URLLC": n_urllc, "mMTC": n_mmtc}

        ue_slices = (
            [("eMBB", i) for i in range(n_embb)]
            + [("URLLC", n_embb + i) for i in range(n_urllc)]
            + [("mMTC", n_embb + n_urllc + i) for i in range(n_mmtc)]
        )
        for slice_type, ue_id in ue_slices:
            self._env.process(
                ue_process(self._env, ue_id, slice_type, self._demand_store, self.rng)
            )

    def step(self) -> dict:
        """Advance simulation by 1 timestep and return demand snapshot."""
        t_now = int(self._env.now)
        self._env.run(until=t_now + 1)
        return {
            "eMBB": round(self._demand_store["eMBB"].get(t_now, 0.0), 4),
            "URLLC": round(self._demand_store["URLLC"].get(t_now, 0.0), 4),
            "mMTC": round(self._demand_store["mMTC"].get(t_now, 0.0), 4),
            "active_users": [
                self._ue_counts["eMBB"],
                self._ue_counts["URLLC"],
                self._ue_counts["mMTC"],
            ],
            "t": t_now,
        }
