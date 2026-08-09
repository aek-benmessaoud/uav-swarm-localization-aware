"""
utils/logger.py — Lightweight passive episode logger (per-step CSV).
All values stored as-is; the logger never computes or aggregates.
"""

import csv
import os


class EpisodeLogger:
    """Buffers per-step data for one episode and writes to CSV on save()."""

    FIELDS = [
        "step", "coverage", "U", "U_norm", "policy_type", "mode",
        "alpha", "F1", "F2", "collision_count",
    ]

    def __init__(self, method, run_label, log_dir="results/logs"):
        self.method = method
        self.run_label = run_label
        safe = method.lower().replace(" ", "_").replace("@", "_")
        self.out_dir = os.path.join(log_dir, safe)
        os.makedirs(self.out_dir, exist_ok=True)
        self.filepath = os.path.join(self.out_dir, f"run_{run_label}.csv")
        self.buffer = []

    def log_step(self, step, coverage, U=None, U_norm=None, policy_type=None,
                 mode=None, alpha=None, F1=None, F2=None, collisions=0):
        self.buffer.append({
            "step": step,
            "coverage": round(coverage, 4),
            "U": round(U, 4) if U is not None else None,
            "U_norm": round(U_norm, 6) if U_norm is not None else None,
            "policy_type": policy_type,
            "mode": mode,
            "alpha": round(alpha, 4) if alpha is not None else None,
            "F1": F1,
            "F2": F2,
            "collision_count": collisions,
        })

    def save(self):
        with open(self.filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writeheader()
            writer.writerows(self.buffer)
        self.buffer = []
