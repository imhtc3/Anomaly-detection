"""Simulate per-minute service metrics with labeled incidents."""
import numpy as np
import pandas as pd

INCIDENT_TYPES = ["cpu_spike", "latency_spike", "error_burst", "traffic_drop", "memory_leak"]


def generate_metrics(days: int = 30, seed: int = 7, n_incidents: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = days * 24 * 60
    ts = pd.date_range("2026-09-01", periods=n, freq="min")
    minute_of_day = np.arange(n) % (24 * 60)
    daily = np.sin(2 * np.pi * (minute_of_day - 6 * 60) / (24 * 60))  # peak mid-afternoon

    requests = 1000 + 400 * daily + rng.normal(0, 40, n)
    cpu = 35 + 15 * daily + rng.normal(0, 3, n)
    latency = 120 + 25 * daily + rng.normal(0, 8, n)
    errors = np.clip(0.5 + rng.normal(0, 0.15, n), 0, None)
    memory = 55 + rng.normal(0, 1.5, n)
    label = np.zeros(n, dtype=int)
    kind = np.array([""] * n, dtype=object)

    starts = rng.choice(np.arange(60, n - 240), size=n_incidents, replace=False)
    for i, s in enumerate(sorted(starts)):
        t = INCIDENT_TYPES[i % len(INCIDENT_TYPES)]
        length = int(rng.integers(5, 30)) if t != "memory_leak" else int(rng.integers(90, 180))
        sl = slice(s, s + length)
        if t == "cpu_spike":
            cpu[sl] += rng.uniform(30, 45)
        elif t == "latency_spike":
            latency[sl] *= rng.uniform(2.5, 4)
        elif t == "error_burst":
            errors[sl] += rng.uniform(4, 10)
        elif t == "traffic_drop":
            requests[sl] *= rng.uniform(0.1, 0.4)
        elif t == "memory_leak":
            memory[sl] += np.linspace(0, rng.uniform(25, 35), length)
        label[sl] = 1
        kind[sl] = t

    return pd.DataFrame({"timestamp": ts, "requests": requests, "cpu_pct": np.clip(cpu, 0, 100),
                         "latency_ms": latency, "error_pct": errors,
                         "memory_pct": np.clip(memory, 0, 100), "is_incident": label,
                         "incident_type": kind})
