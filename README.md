# ⬡ SHC — Self-Healing Cluster

> A Kubernetes-native system that **automatically detects and heals unhealthy nodes** using a Machine Learning anomaly detection model, with a real-time live dashboard.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Features](#features)
4. [Project Structure](#project-structure)
5. [Prerequisites](#prerequisites)
6. [Quick Start (Local — No Kubernetes)](#quick-start-local--no-kubernetes)
7. [Running on Kubernetes](#running-on-kubernetes)
8. [Dashboard Guide](#dashboard-guide)
9. [API Reference](#api-reference)
10. [ML Model Details](#ml-model-details)
11. [Troubleshooting](#troubleshooting)
12. [Tech Stack](#tech-stack)

---

## Overview

SHC (Self-Healing Cluster) monitors cloud node metrics, feeds them to an **Isolation Forest ML model**, and automatically restarts unhealthy pods when a persistent anomaly is confirmed using a **double-verification strategy** (detects → waits 20 s → re-checks → heals). Everything is visualised on a live dark-mode dashboard.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Browser Dashboard                     │
│        (WebSocket · Chart.js · Real-time UI)            │
└───────────────────────┬─────────────────────────────────┘
                        │ WebSocket ws://
                        ▼
┌─────────────────────────────────────────────────────────┐
│               Demo-Service  (Node.js)                   │
│  • 5-second metric broadcast loop                       │
│  • 30-second anomaly check loop                         │
│  • Double-verification before healing                   │
│  • Kubernetes API for pod restarts                      │
└───────────────┬─────────────────────────────────────────┘
                │ POST /predict (HTTP)
                ▼
┌─────────────────────────────────────────────────────────┐
│               ML-Service  (Python FastAPI)              │
│  • Isolation Forest (200 estimators, 8 features)        │
│  • Trained on 5 failure scenario types                  │
│  • Returns { anomaly: true/false, score: float }        │
└─────────────────────────────────────────────────────────┘
```

---

## Features

- 🤖 **ML Anomaly Detection** — Isolation Forest trained on 5 failure types
- 🔁 **Double-Verification** — confirms anomaly before any healing action (no false positives)
- 🩺 **Automatic Pod Restart** — via Kubernetes API with RBAC scoped to minimum permissions
- 📊 **Live Dashboard** — WebSocket-powered real-time metrics, node health map, rolling charts, healing event log
- ⚡ **Demo Controls** — "Simulate Stress" and "Reset" buttons to show the full healing cycle
- 🛡️ **Fallback Threshold** — works even if ML service is temporarily unreachable
- 🐳 **Fully Dockerised** — both services have production-ready Dockerfiles

---

## Project Structure

```
SHC/
├── .gitignore
├── README.md
│
├── Demo-Service/               ← Node.js monitor + dashboard server
│   ├── index.js                   Main application
│   ├── package.json
│   ├── Dockerfile
│   ├── deployment.yaml            Kubernetes Deployment
│   ├── service.yaml               Kubernetes Service (NodePort)
│   ├── rbac.yaml                  ServiceAccount + Role + RoleBinding
│   └── dashboard/
│       ├── index.html             Dashboard UI
│       ├── style.css              Dark glassmorphism styles
│       └── app.js                 WebSocket client + Chart.js logic
│
└── ML-Model/                   ← Python ML anomaly detection service
    ├── train_model.py             Dataset generator + model trainer
    ├── ml_service.py              FastAPI prediction service
    ├── test_model.py              Quick model validation script
    ├── requirements.txt           Python dependencies
    ├── Dockerfile
    ├── ml-deployment.yaml         Kubernetes Deployment
    ├── ml-service.yaml            Kubernetes Service (ClusterIP)
    ├── anomaly_model.pkl          Trained model (generated)
    └── scaler.pkl                 Feature scaler (generated)
```

---

## Prerequisites

Make sure the following are installed on your system:

| Tool | Version | Purpose |
|------|---------|---------|
| **Node.js** | ≥ 18.x | Demo-Service runtime |
| **npm** | ≥ 9.x | Node package manager |
| **Python** | ≥ 3.10 | ML model and service |
| **pip** | ≥ 23.x | Python package manager |

> **For Kubernetes deployment only:**
>
> | Tool | Purpose |
> |------|---------|
> | Docker Desktop / Docker Engine | Build container images |
> | kubectl | Manage Kubernetes cluster |
> | Minikube / Kind / any K8s cluster | The cluster itself |

### Installing Prerequisites

**Node.js** — https://nodejs.org/en/download  
**Python** — https://www.python.org/downloads  
**Docker Desktop** — https://www.docker.com/products/docker-desktop  
**kubectl** — https://kubernetes.io/docs/tasks/tools  
**Minikube** — https://minikube.sigs.k8s.io/docs/start  

---

## Quick Start (Local — No Kubernetes)

This runs everything on your laptop with no Kubernetes needed. Ideal for demos and development.

### Step 1 — Clone / Navigate to the project

```bash
cd SHC
```

### Step 2 — Train the ML Model

```bash
cd ML-Model
pip install -r requirements.txt
python train_model.py
```

Expected output:
```
Generating synthetic node metrics dataset...
Dataset: 5000 total samples  (4500 normal + 500 anomalous)
Model trained. Flagged 500/5000 samples as anomalous (10.0%)
Saved: anomaly_model.pkl  scaler.pkl
```

### Step 3 — Start the ML Service

Keep this terminal open:

```bash
# Still inside ML-Model/
uvicorn ml_service:app --host 0.0.0.0 --port 8000
```

Verify it's up: http://localhost:8000/health → `{"status":"ok"}`

### Step 4 — Start the Demo Service + Dashboard

Open a **new terminal**:

```bash
cd SHC/Demo-Service
npm install
```

**Windows (PowerShell):**
```powershell
$env:ML_SERVICE_URL = "http://localhost:8000"
node index.js
```

**macOS / Linux (bash/zsh):**
```bash
ML_SERVICE_URL=http://localhost:8000 node index.js
```

### Step 5 — Open the Dashboard

Open your browser and go to:

```
http://localhost:3000
```

You should see the live dark-mode dashboard with metrics updating every 5 seconds.

---

## Running on Kubernetes

### Step 1 — Start Minikube

```bash
minikube start
eval $(minikube docker-env)    # macOS/Linux
# Windows PowerShell:
# & minikube -p minikube docker-env --shell powershell | Invoke-Expression
```

### Step 2 — Build Docker Images

```bash
# Build ML service image
cd SHC/ML-Model
docker build -t ml-anomaly-service .

# Build Demo-Service image
cd ../Demo-Service
docker build -t selfheal-app .
```

### Step 3 — Deploy to Kubernetes

```bash
cd SHC

# RBAC (service account, role, rolebinding)
kubectl apply -f Demo-Service/rbac.yaml

# ML Service
kubectl apply -f ML-Model/ml-deployment.yaml
kubectl apply -f ML-Model/ml-service.yaml

# Demo Service + Dashboard
kubectl apply -f Demo-Service/deployment.yaml
kubectl apply -f Demo-Service/service.yaml
```

### Step 4 — Access the Dashboard

```bash
minikube service selfheal-service
```

This opens the dashboard automatically in your browser.

### Step 5 — Verify All Pods are Running

```bash
kubectl get pods
kubectl get services
```

Expected:
```
NAME                            READY   STATUS    RESTARTS
ml-service-xxxx                 1/1     Running   0
selfheal-app-xxxx               1/1     Running   0
```

---

## Dashboard Guide

| Section | Description |
|---------|-------------|
| **Header** | System name, live status badge (NORMAL / DETECTING / CONFIRMED), live clock, WebSocket connection indicator |
| **Cluster Nodes** | 3 animated node cards (Master + 2 Workers) — change colour based on anomaly state |
| **Live Metrics** | 8 metric cards with progress bars — turn amber/red when thresholds exceeded |
| **Rolling Chart** | Chart.js multi-line chart showing last 60 data points for CPU, Memory, Latency, Error Rate |
| **Anomaly Detection** | Pulsing indicator with state description + counters (Heals, Anomalies, Uptime) |
| **Healing Log** | Table of all healing events — timestamp, issue, key metrics, action taken |
| **Demo Controls** | "⚡ Simulate Node Stress" — triggers anomalous metrics immediately |
| | "↺ Reset to Normal" — resets simulation back to normal metrics |
| | Links to raw `/api/metrics` and `/api/events` JSON |

### Demo Scenario for Presentation

1. Open dashboard → show **NORMAL** state, point out all 8 live metrics
2. Click **"⚡ Simulate Node Stress"**
3. Within ~30 seconds:
   - Status badge changes: **NORMAL → DETECTING → CONFIRMED**
   - Node cards change colour: Healthy → Degraded → Critical
   - Metric cards turn red
   - Anomaly count increments
4. After healing is confirmed: new row appears in **Healing Event Log**
5. Click **"↺ Reset"** — system recovers automatically to NORMAL
6. Mention the automatic **5 failure scenario rotation**: CPU Spike → OOM → Disk I/O → Network → Crash Loop

---

## API Reference

All endpoints are on the **Demo-Service** (`http://localhost:3000`):

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the dashboard UI |
| `GET` | `/health` | Health check → `{"status":"ok","state":"NORMAL"}` |
| `GET` | `/api/metrics` | Latest metrics snapshot (JSON) |
| `GET` | `/api/events` | Full healing event log (JSON array) |
| `GET` | `/api/state` | Current state + uptime stats |
| `GET` | `/stress` | Trigger anomalous metrics simulation |
| `GET` | `/reset` | Reset metrics to normal |
| `GET` | `/crash?token=shc-secret` | Intentional crash (token-protected) |

**ML Service** (`http://localhost:8000`):

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | `{"status":"ok"}` |
| `GET` | `/info` | Model metadata (algorithm, features, contamination) |
| `POST` | `/predict` | Predict anomaly from 8 metrics → `{"anomaly":bool,"score":float}` |
| `GET` | `/docs` | Interactive Swagger UI |

---

## ML Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | Isolation Forest |
| Library | scikit-learn |
| Estimators | 200 trees |
| Contamination | 10% |
| Training samples | 5000 (4500 normal + 500 anomalous) |
| Features | 8 |
| Random seed | 42 |

### Features

| Feature | Description | Normal Range | Alert Threshold |
|---------|-------------|--------------|-----------------|
| `cpu_usage` | CPU utilisation % | 10–60% | > 85% |
| `memory_usage` | RAM utilisation % | 30–65% | > 90% |
| `request_rate` | Requests per second | 180–380 | < 50 |
| `latency` | Request latency (ms) | 60–280 | > 1500 |
| `pod_restarts` | Restart count | 0–1 | ≥ 4 |
| `disk_io` | Disk I/O utilisation % | 10–55% | > 87% |
| `network_errors` | Errors per minute | 0–6 | > 45 |
| `error_rate` | Error fraction 0–1 | 0–0.04 | > 0.30 |

### Failure Scenarios Trained

| Scenario | Characteristics |
|----------|----------------|
| CPU Spike | cpu_usage > 87%, high latency, elevated error rate |
| Memory Exhaustion (OOM) | memory_usage > 90%, many pod restarts, high error rate |
| Disk I/O Saturation | disk_io > 88%, extreme latency > 1800ms |
| Network Degradation | network_errors > 48/min, very high latency, high error rate |
| Crash Loop | pod_restarts > 5, high CPU + memory, low request rate |

---

## Troubleshooting

### Port 3000 already in use
```powershell
# Windows
netstat -ano | findstr ":3000"
taskkill /PID <PID> /F

# Or run on a different port:
$env:PORT = "3001"
node index.js
```

### ML service not reachable
The Demo-Service has a **built-in fallback** using fixed thresholds — it will still detect anomalies even without the ML service. Check that `ML_SERVICE_URL` is set correctly.

### `anomaly_model.pkl` not found
Re-run the training script from inside the `ML-Model/` directory:
```bash
cd ML-Model
python train_model.py
```

### Python `pip install` fails
Try using a virtual environment:
```bash
cd ML-Model
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
pip install -r requirements.txt
```

### Minikube image not found
Make sure you built the Docker images **after** running `eval $(minikube docker-env)` so the images exist inside Minikube's registry:
```bash
eval $(minikube docker-env)   # must run this first!
docker build -t ml-anomaly-service ./ML-Model
docker build -t selfheal-app ./Demo-Service
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | Python · scikit-learn (IsolationForest) · pandas · numpy · joblib |
| ML Service | FastAPI · uvicorn · Pydantic |
| Monitor Service | Node.js · Express · ws (WebSocket) · axios |
| Kubernetes Client | @kubernetes/client-node |
| Dashboard | HTML5 · CSS3 · JavaScript (ES2022) · Chart.js |
| Containerisation | Docker |
| Orchestration | Kubernetes · kubectl |
| RBAC | Kubernetes ServiceAccount + Role + RoleBinding |

---

## Author
Swaroop Vyawahare

Final Year Academic Project — Self-Healing Cluster (SHC)

> Built to demonstrate how ML-driven observability can automate Kubernetes node recovery without human intervention.
