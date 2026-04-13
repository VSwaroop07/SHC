"use strict";
const express  = require("express");
const axios    = require("axios");
const http     = require("http");
const path     = require("path");
const { WebSocketServer } = require("ws");

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, "dashboard")));

const PORT           = process.env.PORT            || 3000;
const ML_URL         = process.env.ML_SERVICE_URL  || "http://ml-service:8000";
const CRASH_TOKEN    = process.env.CRASH_TOKEN     || "shc-secret";

// ── State ────────────────────────────────────────────────────────────────────
const MAX_LOG   = 50;
let healingLog  = [];
let lastMetrics = {};
let currentState = "NORMAL";   // NORMAL | DETECTING | CONFIRMED
let monitorBusy  = false;
let simTick      = 0;
let stressMode   = false;
let totalHeals   = 0;
let totalAnomalies = 0;
const startTime  = Date.now();

const SCENARIOS = [
  "CPU Spike",
  "Memory Exhaustion (OOM)",
  "Disk I/O Saturation",
  "Network Degradation",
  "Crash Loop",
];

// ── Helpers ──────────────────────────────────────────────────────────────────
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const rnd   = (lo, hi)    => lo + Math.random() * (hi - lo);
const round2 = v => Math.round(v * 100) / 100;

function getPhase() {
  if (stressMode) return "anomalous";
  const c = simTick % 120;
  if (c < 80)  return "normal";
  if (c < 100) return "degrading";
  return "anomalous";
}

function getScenario() {
  return SCENARIOS[Math.floor(simTick / 120) % SCENARIOS.length];
}

// ── Metric Simulation ─────────────────────────────────────────────────────────
function collectMetrics() {
  simTick++;
  const phase = getPhase();
  let m;

  if (phase === "normal") {
    m = {
      cpu_usage:      clamp(rnd(10, 60),  5, 70),
      memory_usage:   clamp(rnd(30, 65), 15, 68),
      request_rate:   clamp(rnd(180,380), 80, 500),
      latency:        clamp(rnd(60, 280),  40, 320),
      pod_restarts:   Math.random() < 0.05 ? 1 : 0,
      disk_io:        clamp(rnd(10, 55),   5, 60),
      network_errors: clamp(rnd(0, 6),     0, 8),
      error_rate:     clamp(rnd(0, 0.04),  0, 0.05),
    };
  } else if (phase === "degrading") {
    const p = ((simTick % 120) - 80) / 20;
    m = {
      cpu_usage:      clamp(45  + p * 45  + rnd(-5, 10),   30, 95),
      memory_usage:   clamp(60  + p * 30  + rnd(-4, 8),    40, 96),
      request_rate:   clamp(280 - p * 210 + rnd(-20, 30),  15, 350),
      latency:        clamp(200 + p * 1100 + rnd(-50,100), 100, 2500),
      pod_restarts:   Math.floor(p * 5),
      disk_io:        clamp(35  + p * 55  + rnd(-5, 10),   15, 95),
      network_errors: clamp(p   * 55 + rnd(0, 8),           0, 65),
      error_rate:     clamp(p   * 0.65 + rnd(0, 0.05),      0, 0.75),
    };
  } else {
    // Anomalous — rotate scenarios
    const s = SCENARIOS.indexOf(getScenario());
    switch (s) {
      case 0: // CPU Spike
        m = { cpu_usage:rnd(87,99), memory_usage:rnd(55,82), request_rate:rnd(40,160),
              latency:rnd(900,3200), pod_restarts:Math.floor(rnd(1,4)),
              disk_io:rnd(30,70), network_errors:rnd(5,22), error_rate:rnd(0.2,0.65) };
        break;
      case 1: // OOM
        m = { cpu_usage:rnd(45,78), memory_usage:rnd(90,100), request_rate:rnd(20,130),
              latency:rnd(700,2500), pod_restarts:Math.floor(rnd(4,9)),
              disk_io:rnd(30,72), network_errors:rnd(3,18), error_rate:rnd(0.35,0.88) };
        break;
      case 2: // Disk I/O
        m = { cpu_usage:rnd(38,76), memory_usage:rnd(45,78), request_rate:rnd(20,120),
              latency:rnd(1800,5500), pod_restarts:Math.floor(rnd(0,3)),
              disk_io:rnd(88,100), network_errors:rnd(8,28), error_rate:rnd(0.25,0.65) };
        break;
      case 3: // Network Degradation
        m = { cpu_usage:rnd(28,68), memory_usage:rnd(38,72), request_rate:rnd(8,80),
              latency:rnd(1500,5800), pod_restarts:Math.floor(rnd(1,5)),
              disk_io:rnd(20,58), network_errors:rnd(48,100), error_rate:rnd(0.5,0.96) };
        break;
      default: // Crash Loop
        m = { cpu_usage:rnd(58,92), memory_usage:rnd(65,92), request_rate:rnd(8,70),
              latency:rnd(1200,4500), pod_restarts:Math.floor(rnd(5,15)),
              disk_io:rnd(38,80), network_errors:rnd(12,38), error_rate:rnd(0.45,0.90) };
    }
  }

  // Round all values
  Object.keys(m).forEach(k => { m[k] = round2(m[k]); });
  lastMetrics = { ...m, phase, scenario: phase === "anomalous" ? getScenario() : null, ts: Date.now() };
  return m;
}

// ── ML Call ───────────────────────────────────────────────────────────────────
async function checkAnomaly(metrics) {
  try {
    const { data } = await axios.post(`${ML_URL}/predict`, metrics, { timeout: 5000 });
    console.log(`[ML] anomaly=${data.anomaly}  score=${data.score}`);
    return data.anomaly;
  } catch (err) {
    console.warn("[ML] Fallback threshold used:", err.message);
    return (
      metrics.cpu_usage      > 80  ||
      metrics.memory_usage   > 87  ||
      metrics.pod_restarts   >= 4  ||
      metrics.error_rate     > 0.30 ||
      metrics.network_errors > 32  ||
      metrics.disk_io        > 87
    );
  }
}

// ── Kubernetes ────────────────────────────────────────────────────────────────
let k8sApi = null;
try {
  const k8s = require("@kubernetes/client-node");
  const kc  = new k8s.KubeConfig();
  kc.loadFromCluster();
  k8sApi = kc.makeApiClient(k8s.CoreV1Api);
  console.log("[K8s] In-cluster config loaded.");
} catch (_) {
  console.warn("[K8s] Not in-cluster — pod restarts will be simulated.");
}

async function restartPod() {
  if (!k8sApi) {
    console.log("[K8s][SIM] Simulating pod deletion for app=selfheal-app...");
    await new Promise(r => setTimeout(r, 1500));
    console.log("[K8s][SIM] Pod deletion complete.");
    return;
  }
  try {
    const pods = await k8sApi.listNamespacedPod(
      "default", undefined, undefined, undefined, undefined, "app=selfheal-app"
    );
    for (const pod of pods.body.items) {
      console.log("[K8s] Deleting pod:", pod.metadata.name);
      await k8sApi.deleteNamespacedPod(pod.metadata.name, "default");
    }
  } catch (err) {
    console.error("[K8s] Restart failed:", err.body || err.message);
  }
}

// ── WebSocket ─────────────────────────────────────────────────────────────────
const server = http.createServer(app);
const wss    = new WebSocketServer({ server });

function broadcast(payload) {
  const msg = JSON.stringify(payload);
  wss.clients.forEach(c => { if (c.readyState === 1) c.send(msg); });
}

wss.on("connection", ws => {
  console.log("[WS] Client connected");
  ws.send(JSON.stringify({
    type: "init",
    state: currentState,
    metrics: lastMetrics,
    log: healingLog,
    stats: getStats(),
  }));
  ws.on("close", () => console.log("[WS] Client disconnected"));
});

// ── Healing Log ───────────────────────────────────────────────────────────────
function addHealingEvent(metrics, action, issue) {
  totalHeals++;
  const event = {
    id:      Date.now(),
    time:    new Date().toISOString(),
    issue,
    action,
    metrics: { ...metrics },
  };
  healingLog.unshift(event);
  if (healingLog.length > MAX_LOG) healingLog.pop();
  broadcast({ type: "healing", data: event, stats: getStats() });
}

function getStats() {
  const uptimeSec = Math.floor((Date.now() - startTime) / 1000);
  return { totalAnomalies, totalHeals, uptimeSec };
}

// ── Monitor Loop ──────────────────────────────────────────────────────────────
async function monitor() {
  if (monitorBusy) return;
  monitorBusy = true;
  try {
    const metrics = collectMetrics();
    broadcast({ type: "metrics", data: { ...lastMetrics }, state: currentState, stats: getStats() });

    const first = await checkAnomaly(metrics);
    if (!first) {
      currentState = "NORMAL";
      monitorBusy  = false;
      return;
    }

    totalAnomalies++;
    currentState = "DETECTING";
    broadcast({ type: "state", state: "DETECTING", metrics: lastMetrics, stats: getStats() });
    console.log("[MON] Anomaly detected — rechecking in 20 s…");

    await new Promise(r => setTimeout(r, 20000));

    const m2     = collectMetrics();
    const second = await checkAnomaly(m2);

    if (second) {
      currentState = "CONFIRMED";
      broadcast({ type: "state", state: "CONFIRMED", metrics: lastMetrics, stats: getStats() });
      console.log("[MON] Persistent anomaly confirmed — healing…");

      const issue  = lastMetrics.scenario || "Unknown Anomaly";
      await restartPod();
      addHealingEvent(m2, "Pod restarted via Kubernetes API", issue);

      // Reset sim to normal after heal
      simTick    = 0;
      stressMode = false;
      await new Promise(r => setTimeout(r, 2000));
    } else {
      console.log("[MON] Transient spike — no action taken.");
    }

    currentState = "NORMAL";
    broadcast({ type: "state", state: "NORMAL", metrics: lastMetrics, stats: getStats() });
  } catch (err) {
    console.error("[MON] Error:", err.message);
  } finally {
    monitorBusy = false;
  }
}

// Broadcast metrics every 5 s, run anomaly check every 30 s
setInterval(() => {
  const m = collectMetrics();
  broadcast({ type: "metrics", data: { ...lastMetrics }, state: currentState, stats: getStats() });
}, 5000);

setInterval(monitor, 30000);

// Run once at start
setTimeout(monitor, 3000);

// ── HTTP Routes ───────────────────────────────────────────────────────────────
app.get("/health", (_req, res) =>
  res.json({ status: "ok", state: currentState })
);
app.get("/api/metrics", (_req, res) => res.json(lastMetrics));
app.get("/api/events",  (_req, res) => res.json(healingLog));
app.get("/api/state",   (_req, res) => res.json({ state: currentState, phase: getPhase(), stats: getStats() }));

// Trigger stress (safe — no event-loop block)
app.get("/stress", (_req, res) => {
  stressMode = true;
  res.json({ message: "Stress mode activated — metrics entering anomalous range." });
});

// Reset
app.get("/reset", (_req, res) => {
  stressMode   = false;
  simTick      = 0;
  currentState = "NORMAL";
  broadcast({ type: "state", state: "NORMAL", metrics: lastMetrics, stats: getStats() });
  res.json({ message: "System reset to normal." });
});

// Crash — token-protected
app.get("/crash", (req, res) => {
  if (req.query.token !== CRASH_TOKEN) {
    return res.status(401).json({ error: "Unauthorized. Provide ?token=<CRASH_TOKEN>" });
  }
  res.json({ message: "Crashing now…" });
  setTimeout(() => process.exit(1), 500);
});

// ── Start ─────────────────────────────────────────────────────────────────────
server.listen(PORT, () => {
  console.log(`SHC Demo-Service running on http://localhost:${PORT}`);
  console.log(`Dashboard  → http://localhost:${PORT}/`);
  console.log(`ML Service → ${ML_URL}`);
});
