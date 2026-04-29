"""
SHC — test_model.py
Tests the trained Isolation Forest on the full 8-feature dataset.
Regenerates data in-memory using the same seed as train_model.py,
so it works regardless of which CSV version is on disk.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# ── Paths: model files live in the same directory as this script ──
ML_DIR = Path(__file__).parent

model  = joblib.load(ML_DIR / "anomaly_model.pkl")
scaler = joblib.load(ML_DIR / "scaler.pkl")

# ── Reproduce the 8-feature dataset (identical seed to train_model.py) ──
np.random.seed(42)

FEATURES = [
    "cpu_usage", "memory_usage", "request_rate", "latency",
    "pod_restarts", "disk_io", "network_errors", "error_rate",
]

def clamp(df):
    df["cpu_usage"]      = df["cpu_usage"].clip(0, 100)
    df["memory_usage"]   = df["memory_usage"].clip(0, 100)
    df["request_rate"]   = df["request_rate"].clip(0, 1000)
    df["latency"]        = df["latency"].clip(10, 10000)
    df["pod_restarts"]   = df["pod_restarts"].clip(0, 20).round().astype(int)
    df["disk_io"]        = df["disk_io"].clip(0, 100)
    df["network_errors"] = df["network_errors"].clip(0, 100)
    df["error_rate"]     = df["error_rate"].clip(0, 1)
    return df

def gen_normal(n):
    df = pd.DataFrame({
        "cpu_usage":      np.random.normal(38, 15, n),
        "memory_usage":   np.random.normal(50, 13, n),
        "request_rate":   np.random.normal(275, 80, n),
        "latency":        np.random.normal(160, 55, n),
        "pod_restarts":   np.random.choice([0,1], n, p=[0.93, 0.07]).astype(float),
        "disk_io":        np.random.normal(32, 14, n),
        "network_errors": np.abs(np.random.normal(3, 2.5, n)),
        "error_rate":     np.abs(np.random.normal(0.025, 0.015, n)),
    })
    burst_idx = np.random.choice(n, size=int(0.08 * n), replace=False)
    df.loc[burst_idx, "cpu_usage"]  += np.random.uniform(25, 40, len(burst_idx))
    df.loc[burst_idx, "latency"]    += np.random.uniform(200, 500, len(burst_idx))
    df.loc[burst_idx, "error_rate"] += np.random.uniform(0.05, 0.15, len(burst_idx))
    return clamp(df)

def gen_cpu_spike(n):
    return clamp(pd.DataFrame({
        "cpu_usage":      np.random.normal(82, 9, n),
        "memory_usage":   np.random.normal(55, 13, n),
        "request_rate":   np.random.normal(230, 70, n),
        "latency":        np.random.normal(900, 350, n),
        "pod_restarts":   np.random.choice([1,2], n, p=[0.65, 0.35]).astype(float),
        "disk_io":        np.random.normal(35, 14, n),
        "network_errors": np.abs(np.random.normal(5, 3, n)),
        "error_rate":     np.random.normal(0.18, 0.10, n).clip(0.02, 0.6),
    }))

def gen_oom(n):
    t = np.linspace(0, 1, n)
    return clamp(pd.DataFrame({
        "cpu_usage":      np.random.normal(55, 15, n),
        "memory_usage":   65 + 30 * t + np.random.normal(0, 4, n),
        "request_rate":   np.random.normal(200, 65, n),
        "latency":        np.random.normal(600, 250, n),
        "pod_restarts":   np.round(np.random.normal(3, 1.5, n)).clip(0, 10),
        "disk_io":        np.random.normal(38, 13, n),
        "network_errors": np.abs(np.random.normal(4, 2.5, n)),
        "error_rate":     np.random.normal(0.28, 0.12, n).clip(0.05, 0.7),
    }))

def gen_disk_flood(n):
    return clamp(pd.DataFrame({
        "cpu_usage":      np.random.normal(52, 16, n),
        "memory_usage":   np.random.normal(58, 13, n),
        "request_rate":   np.random.normal(180, 60, n),
        "latency":        np.random.normal(1500, 600, n),
        "pod_restarts":   np.random.choice([0,1,2], n, p=[0.5, 0.35, 0.15]).astype(float),
        "disk_io":        np.random.normal(88, 7, n),
        "network_errors": np.abs(np.random.normal(7, 4, n)),
        "error_rate":     np.random.normal(0.22, 0.10, n).clip(0.05, 0.6),
    }))

def gen_network_degradation(n):
    return clamp(pd.DataFrame({
        "cpu_usage":      np.random.normal(42, 15, n),
        "memory_usage":   np.random.normal(52, 13, n),
        "request_rate":   np.random.normal(120, 55, n),
        "latency":        np.random.normal(1200, 500, n),
        "pod_restarts":   np.random.choice([1,2,3], n, p=[0.5, 0.35, 0.15]).astype(float),
        "disk_io":        np.random.normal(33, 13, n),
        "network_errors": np.random.normal(45, 18, n).clip(15, 100),
        "error_rate":     np.random.normal(0.45, 0.15, n).clip(0.1, 0.9),
    }))

def gen_crash_loop(n):
    restarts = np.random.normal(7, 3.5, n).clip(2, 20)
    recovery = np.random.choice(n, size=int(0.30 * n), replace=False)
    restarts[recovery] = np.random.choice([1, 2], len(recovery))
    return clamp(pd.DataFrame({
        "cpu_usage":      np.random.normal(65, 15, n),
        "memory_usage":   np.random.normal(70, 12, n),
        "request_rate":   np.random.normal(90, 45, n),
        "latency":        np.random.normal(1200, 500, n),
        "pod_restarts":   restarts,
        "disk_io":        np.random.normal(45, 15, n),
        "network_errors": np.abs(np.random.normal(12, 6, n)),
        "error_rate":     np.random.normal(0.40, 0.15, n).clip(0.05, 0.9),
    }))

N_NORMAL, N_ANOM = 4500, 500
n_each = N_ANOM // 5

frames = [
    gen_normal(N_NORMAL),
    gen_cpu_spike(n_each),
    gen_oom(n_each),
    gen_disk_flood(n_each),
    gen_network_degradation(n_each),
    gen_crash_loop(N_ANOM - 4 * n_each),
]
labels = [0] * N_NORMAL + [1] * N_ANOM

data = pd.concat(frames, ignore_index=True)
data["label"] = labels
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# ── Scale & Predict ───────────────────────────────────────
X_scaled    = scaler.transform(data[FEATURES].values)
predictions = model.predict(X_scaled)   # +1 = normal, -1 = anomaly

data["prediction"] = predictions
data["anomaly"]    = (predictions == -1).astype(int)

# ── Results ───────────────────────────────────────────────
counts = data["prediction"].value_counts()
print("=" * 40)
print("  SHC — Test Model Results")
print("=" * 40)
print(f"  Total samples : {len(data)}")
print(f"  Normal  (+1)  : {counts.get( 1, 0)}")
print(f"  Anomaly (-1)  : {counts.get(-1, 0)}")
print(f"  Anomaly rate  : {counts.get(-1, 0) / len(data) * 100:.1f}%")
print("=" * 40)

print("\nSample predictions (first 10):")
print(data[FEATURES + ["label", "prediction"]].head(10).to_string(index=False))
