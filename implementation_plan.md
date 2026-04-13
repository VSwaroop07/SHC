# SHC — Full Implementation Plan

## Goal

Transform the existing SHC skeleton into a **fully working, visually demonstrable** Self-Healing Cluster system suitable for a final-year academic project presentation.

The system will:
1. Continuously monitor simulated cloud node metrics
2. Run an ML anomaly detection model (Isolation Forest) trained on realistic node failure scenarios
3. Automatically "heal" nodes when a persistent anomaly is confirmed
4. Show everything live on a **beautiful dark-mode dashboard**

---

## Proposed Changes

### ML-Model

#### [MODIFY] [train_model.py](file:///c:/Users/vyawa/OneDrive/Desktop/SHC/ML-Model/train_model.py)
- Add more diverse failure scenarios to the dataset generator (CPU spike, OOM, disk I/O flood, network degradation, crash loop, thermal throttle)
- Tune `contamination=0.08` to better reflect real-world failure rates

#### [MODIFY] [ml_service.py](file:///c:/Users/vyawa/OneDrive/Desktop/SHC/ML-Model/ml_service.py)
- Replace raw [dict](file:///c:/Users/vyawa/OneDrive/Desktop/SHC/ML-Model/ml_service.py#10-28) parameter with a **Pydantic `MetricsInput` model** (prevents 500 on missing keys)
- Add a `/health` GET endpoint returning `{ "status": "ok" }`
- Add a `/info` GET endpoint that returns model metadata (algorithm, feature names, contamination rate)

#### [MODIFY] [Dockerfile](file:///c:/Users/vyawa/OneDrive/Desktop/SHC/ML-Model/Dockerfile)
- Add a `requirements.txt` and install from it (better practice than inline pip)

#### [MODIFY] [ml-deployment.yaml](file:///c:/Users/vyawa/OneDrive/Desktop/SHC/ML-Model/ml-deployment.yaml)
- Add resource `requests` and `limits` (memory: 256Mi/512Mi, cpu: 100m/300m)

---

### Demo-Service

#### [MODIFY] [index.js](file:///c:/Users/vyawa/OneDrive/Desktop/SHC/Demo-Service/index.js)
- Fix `/stress`: move CPU burn into a `worker_thread` so the event loop stays alive
- Guard `/crash` with a `?token=secret` query param check
- `await restartPod()` in the monitor loop
- Replace pure-random metrics with **trending simulation** (metrics drift into anomalous ranges for demo effect)
- Add in-memory `healingLog[]` (last 50 events) — each entry: `{ time, type, metrics, action }`
- Add **WebSocket server** (`ws` package) — broadcasts live events to the dashboard
- Add REST endpoint `GET /api/events` returning the healing log (for dashboard initial load)
- Add REST endpoint `GET /api/metrics` returning the last collected metrics snapshot
- Serve the `dashboard/` folder as static files at `/`

#### [MODIFY] [package.json](file:///c:/Users/vyawa/OneDrive/Desktop/SHC/Demo-Service/package.json)
- Add `ws` (WebSocket) dependency

#### [MODIFY] [deployment.yaml](file:///c:/Users/vyawa/OneDrive/Desktop/SHC/Demo-Service/deployment.yaml)
- Remove the erroneous `replicas` field inside `spec.template.spec`

---

### Dashboard (NEW)

#### [NEW] [dashboard/index.html](file:///c:/Users/vyawa/OneDrive/Desktop/SHC/Demo-Service/dashboard/index.html)
Dark-mode, glassmorphism-style single-page dashboard with:
- **Header**: project name, live clock, cluster status badge
- **Metric cards** (5 cards): CPU, Memory, Request Rate, Latency, Pod Restarts — each with a sparkline mini-chart
- **Main chart**: rolling 60-second line chart of all metrics (Chart.js)
- **Anomaly status panel**: shows NORMAL / DETECTING / ANOMALY CONFIRMED with animated indicator
- **Healing Log table**: timestamp, anomaly type, metrics at detection, action taken
- **Node map**: 3 animated node cards showing health status (Healthy / Degraded / Restarting)
- WebSocket client that connects back to Demo-Service for live updates

#### [NEW] [dashboard/style.css](file:///c:/Users/vyawa/OneDrive/Desktop/SHC/Demo-Service/dashboard/style.css)
- Dark theme (`#0d0f14` background), glassmorphism cards
- Gradient accent colors (cyan/purple)
- Smooth CSS animations for state transitions
- Google Fonts (Inter)

#### [NEW] [dashboard/app.js](file:///c:/Users/vyawa/OneDrive/Desktop/SHC/Demo-Service/dashboard/app.js)
- WebSocket client managing reconnects
- Chart.js setup for all charts
- DOM update functions for metrics, anomaly state, healing log

---

### Kubernetes & Project

#### [MODIFY] [.gitignore](file:///c:/Users/vyawa/OneDrive/Desktop/SHC/.gitignore)
- Add proper entries: `node_modules/`, `*.pkl`, `__pycache__/`, `.env`

#### [NEW] [ML-Model/requirements.txt](file:///c:/Users/vyawa/OneDrive/Desktop/SHC/ML-Model/requirements.txt)
- Pin all Python dependencies

---

## Verification Plan

### Automated / Script Tests

1. **ML model smoke test** — already exists as [test_model.py](file:///c:/Users/vyawa/OneDrive/Desktop/SHC/ML-Model/test_model.py):
   ```bash
   cd ML-Model
   python test_model.py
   ```
   Expected: prints prediction `value_counts()` with some `-1` anomalies present.

2. **FastAPI `/predict` test** — run locally:
   ```bash
   cd ML-Model
   uvicorn ml_service:app --host 0.0.0.0 --port 8000
   # In another terminal:
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"cpu_usage": 95, "memory_usage": 92, "request_rate": 10, "latency": 5000, "pod_restarts": 8, "disk_io": 95, "network_errors": 50, "error_rate": 0.9}'
   ```
   Expected: `{"anomaly": true}` or `{"anomaly": false}`.

3. **Demo-Service API test** — run locally:
   ```bash
   cd Demo-Service
   npm install
   node index.js
   # In another terminal:
   curl http://localhost:3000/api/metrics
   curl http://localhost:3000/api/events
   ```
   Expected: JSON metric snapshots and (initially empty) event array.

### Manual / Browser Verification

4. **Dashboard visual check**: Open `http://localhost:3000` in a browser — verify dark dashboard loads, metric values update every 5 seconds, charts animate smoothly.

5. **Anomaly simulation**: Hit `http://localhost:3000/stress` in a separate tab — observe the dashboard's anomaly indicator change from NORMAL → DETECTING → ANOMALY CONFIRMED within ~50 seconds, and a new row appear in the Healing Log.

6. **WebSocket live feed**: Open browser DevTools → Network → WS — verify the WebSocket connection is active and receives JSON frames every 5 seconds.
