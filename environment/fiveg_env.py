"""
fiveg_env.py — 5G Network Environment: slice isolation, QoS metrics, and SLA checking.

This module bridges raw RL actions (bandwidth fractions) into realistic
network-layer outcomes (throughput, latency, PDR) and reports SLA compliance.
"""

import numpy as np
from config import (
    TOTAL_BANDWIDTH_MHZ,
    SLA_THROUGHPUT_EMBB,
    SLA_LATENCY_URLLC_MS,
    SLA_PDR_MMTC,
)


class FiveGEnvironment:
    """
    Simulates 5G RAN/Core network constraints.

    Responsibilities:
      - Enforce slice bandwidth isolation
      - Compute per-slice QoS metrics
      - Check SLA compliance
    """

    def __init__(self, total_bw: float = TOTAL_BANDWIDTH_MHZ):
        self.total_bw = total_bw

        # Internal state
        self.last_allocation = {"eMBB": 0.5, "URLLC": 0.3, "mMTC": 0.2}
        self.last_demands    = {"eMBB": 0.0, "URLLC": 0.0, "mMTC": 0.0}
        self.last_qos        = {"eMBB": 0.0, "URLLC": 0.0, "mMTC": 0.0}
        self.last_sla_ok     = [True, True, True]

    # ─────────────────────────────────────────
    # Core: allocate and compute QoS
    # ─────────────────────────────────────────

    def allocate(self, demands: dict[str, float], action: np.ndarray) -> dict:
        """
        Apply bandwidth allocation and compute QoS metrics.

        Args:
            demands : {'eMBB': float, 'URLLC': float, 'mMTC': float}  (Mbps)
            action  : np.ndarray of shape (3,) — softmax allocation fractions

        Returns:
            result dict with throughput, latency, pdr, sla_ok, utilization
        """
        # Normalise action fractions (ensure they sum to 1)
        action = np.clip(action, 1e-6, None)
        fracs  = action / action.sum()

        alloc_bw = {
            "eMBB":  fracs[0] * self.total_bw,
            "URLLC": fracs[1] * self.total_bw,
            "mMTC":  fracs[2] * self.total_bw,
        }

        self.last_allocation = {"eMBB": fracs[0], "URLLC": fracs[1], "mMTC": fracs[2]}
        self.last_demands    = demands

        # ── eMBB: throughput ────────────────
        # Available bandwidth vs requested demand (demand in Mbps, bw in MHz)
        # Spectrum efficiency ≈ 5 bits/s/Hz → capacity = bw_MHz * 5 Mbps
        cap_embb = alloc_bw["eMBB"] * 5.0         # Mbps
        demand_embb = demands.get("eMBB", 0.0)
        tput_embb = min(demand_embb, cap_embb)     # actual throughput

        # ── URLLC: latency ──────────────────
        # Model: latency increases as allocation drops below demand
        cap_urllc   = alloc_bw["URLLC"] * 5.0
        demand_urllc = demands.get("URLLC", 0.0)
        load_ratio   = demand_urllc / (cap_urllc + 1e-9)
        # Queuing model: latency ~ 1 / (1 - rho) for rho < 1, capped at 10ms
        rho          = min(load_ratio, 0.999)
        latency_ms   = (0.1 / (1.0 - rho + 1e-9))        # base 0.1 ms at zero load
        latency_ms   = float(np.clip(latency_ms, 0.05, 10.0))

        # ── mMTC: packet delivery ratio ─────
        cap_mmtc    = alloc_bw["mMTC"] * 5.0
        demand_mmtc = demands.get("mMTC", 0.0)
        pdr_mmtc    = float(np.clip(cap_mmtc / (demand_mmtc + 1e-9), 0.0, 1.0))

        # ── Store QoS ───────────────────────
        self.last_qos = {
            "eMBB_throughput_Mbps": round(tput_embb, 4),
            "URLLC_latency_ms":     round(latency_ms, 4),
            "mMTC_PDR":             round(pdr_mmtc, 4),
        }

        # ── SLA Check ───────────────────────
        sla_embb  = tput_embb  >= SLA_THROUGHPUT_EMBB
        sla_urllc = latency_ms <= SLA_LATENCY_URLLC_MS
        sla_mmtc  = pdr_mmtc   >= SLA_PDR_MMTC
        self.last_sla_ok = [sla_embb, sla_urllc, sla_mmtc]

        # ── Utilization (fraction of allocated bw actually used) ─
        util = {
            "eMBB":  round(tput_embb    / (cap_embb  + 1e-9), 4),
            "URLLC": round(min(load_ratio, 1.0), 4),
            "mMTC":  round(demand_mmtc  / (cap_mmtc  + 1e-9), 4),
        }

        return {
            "allocation_fracs": {"eMBB": fracs[0], "URLLC": fracs[1], "mMTC": fracs[2]},
            "allocation_mhz":   alloc_bw,
            "qos":              self.last_qos,
            "sla_ok":           self.last_sla_ok,
            "utilization":      util,
        }

    # ─────────────────────────────────────────
    # QoS score for reward
    # ─────────────────────────────────────────

    def qos_score(self) -> float:
        """
        Scalar QoS reward in [0, 1].
        Combines normalised throughput, latency, and PDR.
        """
        qos = self.last_qos

        # eMBB: normalise throughput (target = SLA_THROUGHPUT_EMBB)
        embb_score  = min(qos["eMBB_throughput_Mbps"] / SLA_THROUGHPUT_EMBB, 1.0)

        # URLLC: normalise latency (target ≤ 1 ms → score = 1 if ≤ 0.5ms)
        urllc_score = float(np.clip(
            1.0 - (qos["URLLC_latency_ms"] - 0.1) / SLA_LATENCY_URLLC_MS, 0.0, 1.0
        ))

        # mMTC: PDR directly as score
        mmtc_score  = qos["mMTC_PDR"]

        # Weighted average (URLLC weighted higher due to strict SLA)
        score = 0.30 * embb_score + 0.50 * urllc_score + 0.20 * mmtc_score
        return round(float(score), 6)

    def check_sla(self) -> list[bool]:
        """Return last computed SLA status [eMBB_ok, URLLC_ok, mMTC_ok]."""
        return self.last_sla_ok
