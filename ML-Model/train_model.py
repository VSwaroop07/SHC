"""
SHC — Realistic Anomaly Detection Dataset Generator + Model Trainer
====================================================================
Changes from original train_model.py:
  1. Reduced separation between normal/anomaly distributions → overlap
  2. Higher within-class noise on all features
  3. Partial anomalies: only 2–4 features deviate, rest stay normal
  4. Gradual/slow-burn anomalies (drift instead of instant spike)
  5. Simulated real-world bursts in NORMAL data (brief cpu/latency spikes)
  6. Saves all 8 features + ground-truth label to CSV for proper evaluation
  7. Trains on 80% → evaluates on 20% held-out test set
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

np.random.seed(42)

FEATURES = [
    'cpu_usage', 'memory_usage', 'request_rate', 'latency',
    'pod_restarts', 'disk_io', 'network_errors', 'error_rate'
]

# ── Helper ────────────────────────────────────────────────
def clamp(df):
    df['cpu_usage']      = df['cpu_usage'].clip(0, 100)
    df['memory_usage']   = df['memory_usage'].clip(0, 100)
    df['request_rate']   = df['request_rate'].clip(0, 1000)
    df['latency']        = df['latency'].clip(10, 10000)
    df['pod_restarts']   = df['pod_restarts'].clip(0, 20).round().astype(int)
    df['disk_io']        = df['disk_io'].clip(0, 100)
    df['network_errors'] = df['network_errors'].clip(0, 100)
    df['error_rate']     = df['error_rate'].clip(0, 1)
    return df


def add_noise(arr, scale=0.05):
    """Add proportional Gaussian jitter to a numpy array."""
    return arr + np.random.normal(0, scale * np.abs(arr).mean(), arr.shape)

# ── Normal Traffic ────────────────────────────────────────
def gen_normal(n):
    df = pd.DataFrame({
        'cpu_usage':      np.random.normal(38, 15, n),     # wider SD → overlap
        'memory_usage':   np.random.normal(50, 13, n),
        'request_rate':   np.random.normal(275, 80, n),
        'latency':        np.random.normal(160, 55, n),
        'pod_restarts':   np.random.choice([0,1], n, p=[0.93, 0.07]).astype(float),
        'disk_io':        np.random.normal(32, 14, n),
        'network_errors': np.abs(np.random.normal(3, 2.5, n)),
        'error_rate':     np.abs(np.random.normal(0.025, 0.015, n)),
    })
    # Real-world bursts: ~8% of normal samples get a brief cpu/latency spike
    burst_idx = np.random.choice(n, size=int(0.08 * n), replace=False)
    df.loc[burst_idx, 'cpu_usage']  += np.random.uniform(25, 40, len(burst_idx))
    df.loc[burst_idx, 'latency']    += np.random.uniform(200, 500, len(burst_idx))
    df.loc[burst_idx, 'error_rate'] += np.random.uniform(0.05, 0.15, len(burst_idx))
    return clamp(df)

# ── Anomaly Type 1: CPU Spike (PARTIAL — only cpu + latency abnormal) ──
def gen_cpu_spike(n):
    """Only cpu_usage and latency deviate; memory/disk stay near-normal."""
    df = pd.DataFrame({
        'cpu_usage':      np.random.normal(82, 9, n),       # overlaps normal bursts
        'memory_usage':   np.random.normal(55, 13, n),      # near-normal
        'request_rate':   np.random.normal(230, 70, n),     # slightly lower
        'latency':        np.random.normal(900, 350, n),    # high but noisy
        'pod_restarts':   np.random.choice([1,2], n, p=[0.65, 0.35]).astype(float),
        'disk_io':        np.random.normal(35, 14, n),      # normal
        'network_errors': np.abs(np.random.normal(5, 3, n)),
        'error_rate':     np.random.normal(0.18, 0.10, n).clip(0.02, 0.6),
    })
    return clamp(df)

# ── Anomaly Type 2: Memory Leak / OOM (GRADUAL — slow drift) ─────────
def gen_oom(n):
    """Memory drifts from 65 → 95 over time; other features mostly normal."""
    t  = np.linspace(0, 1, n)   # simulate time progression
    df = pd.DataFrame({
        'cpu_usage':      np.random.normal(55, 15, n),
        'memory_usage':   65 + 30 * t + np.random.normal(0, 4, n),  # gradual drift
        'request_rate':   np.random.normal(200, 65, n),
        'latency':        np.random.normal(600, 250, n),
        'pod_restarts':   np.round(np.random.normal(3, 1.5, n)).clip(0, 10),
        'disk_io':        np.random.normal(38, 13, n),               # near-normal
        'network_errors': np.abs(np.random.normal(4, 2.5, n)),
        'error_rate':     np.random.normal(0.28, 0.12, n).clip(0.05, 0.7),
    })
    return clamp(df)

# ── Anomaly Type 3: Disk I/O Flood (PARTIAL — only disk + latency bad) ─
def gen_disk_flood(n):
    """disk_io saturated; cpu/memory stay near-normal (partial anomaly)."""
    df = pd.DataFrame({
        'cpu_usage':      np.random.normal(52, 16, n),      # near-normal
        'memory_usage':   np.random.normal(58, 13, n),      # near-normal
        'request_rate':   np.random.normal(180, 60, n),
        'latency':        np.random.normal(1500, 600, n),   # high I/O wait
        'pod_restarts':   np.random.choice([0,1,2], n, p=[0.5, 0.35, 0.15]).astype(float),
        'disk_io':        np.random.normal(88, 7, n),       # elevated but noisy
        'network_errors': np.abs(np.random.normal(7, 4, n)),
        'error_rate':     np.random.normal(0.22, 0.10, n).clip(0.05, 0.6),
    })
    return clamp(df)

# ── Anomaly Type 4: Network Degradation (SUBTLE — only net features bad) ─
def gen_network_degradation(n):
    """High network_errors + error_rate only; cpu/memory/disk look normal."""
    df = pd.DataFrame({
        'cpu_usage':      np.random.normal(42, 15, n),      # normal
        'memory_usage':   np.random.normal(52, 13, n),      # normal
        'request_rate':   np.random.normal(120, 55, n),     # low (packets lost)
        'latency':        np.random.normal(1200, 500, n),
        'pod_restarts':   np.random.choice([1,2,3], n, p=[0.5, 0.35, 0.15]).astype(float),
        'disk_io':        np.random.normal(33, 13, n),      # normal
        'network_errors': np.random.normal(45, 18, n).clip(15, 100),  # key signal
        'error_rate':     np.random.normal(0.45, 0.15, n).clip(0.1, 0.9),
    })
    return clamp(df)

# ── Anomaly Type 5: Crash Loop (NOISY — intermittent, hard to detect) ──
def gen_crash_loop(n):
    """Pod restarts high but fluctuating; some samples look nearly normal."""
    restarts = np.random.normal(7, 3.5, n).clip(2, 20)
    # ~30% of crash-loop samples are in a "brief recovery" and look near-normal
    recovery_idx = np.random.choice(n, size=int(0.30 * n), replace=False)
    restarts[recovery_idx] = np.random.choice([1, 2], len(recovery_idx))
    df = pd.DataFrame({
        'cpu_usage':      np.random.normal(65, 15, n),
        'memory_usage':   np.random.normal(70, 12, n),
        'request_rate':   np.random.normal(90, 45, n),
        'latency':        np.random.normal(1200, 500, n),
        'pod_restarts':   restarts,
        'disk_io':        np.random.normal(45, 15, n),
        'network_errors': np.abs(np.random.normal(12, 6, n)),
        'error_rate':     np.random.normal(0.40, 0.15, n).clip(0.05, 0.9),
    })
    return clamp(df)


# ── Generate Dataset ──────────────────────────────────────
N_NORMAL = 4500
N_ANOM   = 500
n_each   = N_ANOM // 5

print("Generating realistic synthetic dataset...")

normal_df        = gen_normal(N_NORMAL);        normal_df['label'] = 0
cpu_df           = gen_cpu_spike(n_each);       cpu_df['label']    = 1
oom_df           = gen_oom(n_each);             oom_df['label']    = 1
disk_df          = gen_disk_flood(n_each);      disk_df['label']   = 1
net_df           = gen_network_degradation(n_each); net_df['label'] = 1
crash_df         = gen_crash_loop(N_ANOM - 4 * n_each); crash_df['label'] = 1

data = pd.concat([normal_df, cpu_df, oom_df, disk_df, net_df, crash_df],
                 ignore_index=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Dataset: {len(data)} samples  "
      f"({(data['label']==0).sum()} normal + {(data['label']==1).sum()} anomalous)")
print("\nFeature statistics:")
print(data[FEATURES].describe().round(2))

# ── Save full dataset (all 8 features + label) ───────────
data.to_csv('self_healing_dataset.csv', index=False)
print("\nSaved: self_healing_dataset.csv  (all 8 features + label)")

# ── Train / Test split (80 / 20, stratified) ─────────────
X = data[FEATURES].values
y = data['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
print(f"\nTrain: {len(X_train)} samples  |  Test: {len(X_test)} samples")

# ── Scale ─────────────────────────────────────────────────
scaler   = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── Train Isolation Forest ────────────────────────────────
# contamination = true anomaly ratio (10%)
model = IsolationForest(
    n_estimators=300,
    contamination=0.10,
    max_features=0.8,       # feature sub-sampling → more robust
    max_samples='auto',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled)

# Quick sanity check on test set
preds_test = model.predict(X_test_scaled)
flagged    = (preds_test == -1).sum()
print(f"Test-set flagged: {flagged}/{len(X_test)} "
      f"({100*flagged/len(X_test):.1f}%)")

# ── Save model & scaler ───────────────────────────────────
joblib.dump(model,  'anomaly_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Saved: anomaly_model.pkl  scaler.pkl")
