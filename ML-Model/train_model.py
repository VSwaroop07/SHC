"""
SHC — Train Anomaly Detection Model
Features: cpu_usage, memory_usage, request_rate, latency,
          pod_restarts, disk_io, network_errors, error_rate
Scenarios: Normal, CPU Spike, OOM, Disk I/O Flood,
           Network Degradation, Crash Loop
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

np.random.seed(42)

FEATURES = [
    'cpu_usage', 'memory_usage', 'request_rate', 'latency',
    'pod_restarts', 'disk_io', 'network_errors', 'error_rate'
]

def clamp_df(df):
    df['cpu_usage']      = df['cpu_usage'].clip(0, 100)
    df['memory_usage']   = df['memory_usage'].clip(0, 100)
    df['request_rate']   = df['request_rate'].clip(0, 1000)
    df['latency']        = df['latency'].clip(10, 10000)
    df['pod_restarts']   = df['pod_restarts'].clip(0, 20).astype(int)
    df['disk_io']        = df['disk_io'].clip(0, 100)
    df['network_errors'] = df['network_errors'].clip(0, 100)
    df['error_rate']     = df['error_rate'].clip(0, 1)
    return df

def gen_normal(n):
    return clamp_df(pd.DataFrame({
        'cpu_usage':      np.random.normal(35, 12, n),
        'memory_usage':   np.random.normal(48, 10, n),
        'request_rate':   np.random.normal(280, 60, n),
        'latency':        np.random.normal(150, 45, n),
        'pod_restarts':   np.random.choice([0, 1], n, p=[0.95, 0.05]).astype(float),
        'disk_io':        np.random.normal(30, 12, n),
        'network_errors': np.abs(np.random.normal(2, 1.5, n)),
        'error_rate':     np.abs(np.random.normal(0.02, 0.01, n)),
    }))

def gen_cpu_spike(n):
    return clamp_df(pd.DataFrame({
        'cpu_usage':      np.random.normal(91, 5, n),
        'memory_usage':   np.random.normal(65, 8, n),
        'request_rate':   np.random.normal(95, 35, n),
        'latency':        np.random.normal(1300, 300, n),
        'pod_restarts':   np.random.choice([1, 2, 3], n, p=[0.5, 0.3, 0.2]).astype(float),
        'disk_io':        np.random.normal(42, 15, n),
        'network_errors': np.abs(np.random.normal(9, 4, n)),
        'error_rate':     np.random.normal(0.32, 0.1, n).clip(0.1, 0.8),
    }))

def gen_oom(n):
    return clamp_df(pd.DataFrame({
        'cpu_usage':      np.random.normal(58, 14, n),
        'memory_usage':   np.random.normal(95, 3, n),
        'request_rate':   np.random.normal(75, 28, n),
        'latency':        np.random.normal(950, 220, n),
        'pod_restarts':   np.random.choice([3,4,5,6,7], n, p=[0.2,0.3,0.25,0.15,0.1]).astype(float),
        'disk_io':        np.random.normal(45, 12, n),
        'network_errors': np.abs(np.random.normal(6, 3, n)),
        'error_rate':     np.random.normal(0.48, 0.14, n).clip(0.2, 0.9),
    }))

def gen_disk_flood(n):
    return clamp_df(pd.DataFrame({
        'cpu_usage':      np.random.normal(54, 14, n),
        'memory_usage':   np.random.normal(60, 12, n),
        'request_rate':   np.random.normal(65, 25, n),
        'latency':        np.random.normal(2100, 550, n),
        'pod_restarts':   np.random.choice([0,1,2], n, p=[0.4, 0.4, 0.2]).astype(float),
        'disk_io':        np.random.normal(93, 4, n),
        'network_errors': np.abs(np.random.normal(11, 5, n)),
        'error_rate':     np.random.normal(0.36, 0.12, n).clip(0.15, 0.7),
    }))

def gen_network_degradation(n):
    return clamp_df(pd.DataFrame({
        'cpu_usage':      np.random.normal(44, 14, n),
        'memory_usage':   np.random.normal(54, 12, n),
        'request_rate':   np.random.normal(38, 20, n),
        'latency':        np.random.normal(1900, 600, n),
        'pod_restarts':   np.random.choice([1,2,3,4], n, p=[0.3,0.35,0.25,0.1]).astype(float),
        'disk_io':        np.random.normal(35, 12, n),
        'network_errors': np.random.normal(65, 20, n).clip(30, 100),
        'error_rate':     np.random.normal(0.62, 0.18, n).clip(0.25, 0.98),
    }))

def gen_crash_loop(n):
    return clamp_df(pd.DataFrame({
        'cpu_usage':      np.random.normal(70, 12, n),
        'memory_usage':   np.random.normal(76, 10, n),
        'request_rate':   np.random.normal(28, 14, n),
        'latency':        np.random.normal(1700, 450, n),
        'pod_restarts':   np.random.normal(8, 3, n).clip(5, 20),
        'disk_io':        np.random.normal(50, 14, n),
        'network_errors': np.abs(np.random.normal(17, 7, n)),
        'error_rate':     np.random.normal(0.55, 0.14, n).clip(0.3, 0.95),
    }))

# ── Generate Dataset ──────────────────────────────────────
N_NORMAL  = 4500
N_ANOM    = 500
n_each    = N_ANOM // 5

print("Generating synthetic node metrics dataset...")

frames = [
    gen_normal(N_NORMAL),
    gen_cpu_spike(n_each),
    gen_oom(n_each),
    gen_disk_flood(n_each),
    gen_network_degradation(n_each),
    gen_crash_loop(N_ANOM - 4 * n_each),
]

data = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Dataset: {len(data)} total samples  ({N_NORMAL} normal + {N_ANOM} anomalous)\n")
print(data[FEATURES].describe().round(2))

# ── Train ─────────────────────────────────────────────────
X = data[FEATURES].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest(
    n_estimators=200,
    contamination=0.10,
    max_features=1.0,
    random_state=42,
    n_jobs=-1
)
model.fit(X_scaled)

preds = model.predict(X_scaled)
flagged = (preds == -1).sum()
print(f"\nModel trained. Flagged {flagged}/{len(data)} samples as anomalous ({100*flagged/len(data):.1f}%)")

# ── Save ──────────────────────────────────────────────────
joblib.dump(model,  'anomaly_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Saved: anomaly_model.pkl  scaler.pkl")
