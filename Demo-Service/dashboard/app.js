"use strict";

// ── Config ───────────────────────────────────────────────────────────────────
const WS_URL      = `ws://${location.host}`;
const MAX_POINTS  = 60;
const RECONNECT_MS = 3000;

// ── Metric definitions ────────────────────────────────────────────────────────
const METRICS = [
  { key: "cpu_usage",      label: "CPU Usage",       unit: "%",      max: 100, warnAt: 70, alertAt: 85, color: "#00d9ff" },
  { key: "memory_usage",   label: "Memory Usage",    unit: "%",      max: 100, warnAt: 75, alertAt: 90, color: "#a855f7" },
  { key: "request_rate",   label: "Request Rate",    unit: "req/s",  max: 500, warnAt: 400, alertAt: 480, color: "#22c55e" },
  { key: "latency",        label: "Latency",         unit: "ms",     max: 5000, warnAt: 600, alertAt: 1500, color: "#f59e0b" },
  { key: "pod_restarts",   label: "Pod Restarts",    unit: "",       max: 15,  warnAt: 2, alertAt: 4, color: "#ff7070" },
  { key: "disk_io",        label: "Disk I/O",        unit: "%",      max: 100, warnAt: 70, alertAt: 87, color: "#fb923c" },
  { key: "network_errors", label: "Network Errors",  unit: "/min",   max: 100, warnAt: 20, alertAt: 45, color: "#e879f9" },
  { key: "error_rate",     label: "Error Rate",      unit: "%",      max: 100, warnAt: 10, alertAt: 30, color: "#f43f5e",
    transform: v => round2(v * 100) },
];

// Chart datasets — only 4 on the main chart to keep it readable
const CHART_METRICS = ["cpu_usage", "memory_usage", "latency", "error_rate"];

function round2(v){ return Math.round(v * 100) / 100; }

// ── Nodes ─────────────────────────────────────────────────────────────────────
const NODES = [
  { id: "node-master", name: "Master Node",   role: "Control Plane", icon: "🖥️" },
  { id: "node-worker1", name: "Worker Node A", role: "Compute",       icon: "⚙️" },
  { id: "node-worker2", name: "Worker Node B", role: "Compute",       icon: "⚙️" },
];

// ── State ─────────────────────────────────────────────────────────────────────
let ws         = null;
let chart      = null;
let labels     = [];
let datasets   = {};
let currentState = "NORMAL";
let rowCounter = 0;

// ── Boot ──────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  buildNodeCards();
  buildMetricCards();
  buildChart();
  startClock();
  connect();
});

// ── Clock ─────────────────────────────────────────────────────────────────────
function startClock() {
  const el = document.getElementById("live-clock");
  const tick = () => {
    const now = new Date();
    el.textContent = now.toLocaleTimeString("en-GB");
  };
  tick();
  setInterval(tick, 1000);
}

// ── WebSocket ─────────────────────────────────────────────────────────────────
function connect() {
  ws = new WebSocket(WS_URL);

  ws.addEventListener("open", () => {
    document.getElementById("ws-dot").classList.add("connected");
    document.getElementById("ws-label").textContent = "live";
  });

  ws.addEventListener("close", () => {
    document.getElementById("ws-dot").classList.remove("connected");
    document.getElementById("ws-label").textContent = "reconnecting…";
    setTimeout(connect, RECONNECT_MS);
  });

  ws.addEventListener("error", () => ws.close());

  ws.addEventListener("message", e => {
    try { handleMessage(JSON.parse(e.data)); }
    catch(err) { console.error("WS parse error:", err); }
  });
}

function handleMessage(msg) {
  switch (msg.type) {
    case "init":
      setState(msg.state);
      if (msg.metrics && msg.metrics.cpu_usage != null) updateMetricCards(msg.metrics);
      if (Array.isArray(msg.log)) msg.log.slice().reverse().forEach(addLogRow);
      if (msg.stats) updateStats(msg.stats);
      break;
    case "metrics":
      if (msg.data && msg.data.cpu_usage != null) {
        updateMetricCards(msg.data);
        pushToChart(msg.data);
      }
      if (msg.state) setState(msg.state);
      if (msg.stats) updateStats(msg.stats);
      break;
    case "state":
      setState(msg.state);
      if (msg.stats) updateStats(msg.stats);
      break;
    case "healing":
      addLogRow(msg.data);
      if (msg.stats) updateStats(msg.stats);
      break;
  }
}

// ── State / Anomaly Panel ────────────────────────────────────────────────────
const STATE_DESC = {
  NORMAL:    "All systems operational",
  DETECTING: "Anomaly suspected — running second verification…",
  CONFIRMED: "Persistent anomaly confirmed — healing pod…",
};

function setState(state) {
  if (state === currentState) return;
  currentState = state;

  // Header badge
  const badge = document.getElementById("cluster-status");
  badge.className = `status-badge ${state}`;
  document.getElementById("status-text").textContent = state;

  // Anomaly indicator
  const ind = document.getElementById("anomaly-indicator");
  ind.className = `anomaly-indicator ${state}`;
  document.getElementById("anom-state").textContent = state;
  document.getElementById("anom-desc").textContent  = STATE_DESC[state] || "";

  // Node card states
  updateNodeStates(state);

  // Scenario label
  const sl = document.getElementById("scenario-label");
  if (state !== "NORMAL") { sl.classList.add("visible"); }
  else                    { sl.classList.remove("visible"); sl.textContent = ""; }
}

function updateNodeStates(state) {
  const cards = document.querySelectorAll(".node-card");
  cards.forEach((card, i) => {
    const pill = card.querySelector(".node-status-pill");
    if (i === 0) {
      // Master always healthy
      card.className = "node-card healthy";
      if (pill) pill.textContent = "● Healthy";
    } else if (state === "NORMAL") {
      card.className = "node-card healthy";
      if (pill) pill.textContent = "● Healthy";
    } else if (state === "DETECTING") {
      card.className = "node-card degraded";
      if (pill) pill.textContent = "⚠ Degraded";
    } else {
      // Only one worker is "critical", the other is degraded
      card.className = i === 1 ? "node-card critical" : "node-card degraded";
      if (pill) pill.textContent = i === 1 ? "✖ Critical" : "⚠ Degraded";
    }
  });
}

// ── Node Cards ────────────────────────────────────────────────────────────────
function buildNodeCards() {
  const grid = document.getElementById("node-grid");
  grid.innerHTML = NODES.map((n, i) => `
    <div class="node-card healthy" id="${n.id}">
      <div class="node-icon-wrap">${n.icon}</div>
      <div class="node-info">
        <span class="node-name">${n.name}</span>
        <span class="node-role">${n.role}</span>
      </div>
      <span class="node-status-pill">● Healthy</span>
    </div>
  `).join("");
}

// ── Metric Cards ──────────────────────────────────────────────────────────────
function buildMetricCards() {
  const grid = document.getElementById("metric-grid");
  grid.innerHTML = METRICS.map(m => `
    <div class="metric-card" id="mc-${m.key}">
      <div class="metric-label">${m.label}</div>
      <div class="metric-value" id="mv-${m.key}" style="color:${m.color}">--</div>
      <div class="metric-unit">${m.unit}</div>
      <div class="metric-bar-wrap">
        <div class="metric-bar" id="mb-${m.key}" style="width:0%;background:${m.color}"></div>
      </div>
    </div>
  `).join("");
}

function updateMetricCards(data) {
  METRICS.forEach(m => {
    let raw = data[m.key];
    if (raw == null) return;
    let val = m.transform ? m.transform(raw) : raw;
    let pct = Math.min(100, (raw / m.max) * 100);

    const valEl  = document.getElementById(`mv-${m.key}`);
    const barEl  = document.getElementById(`mb-${m.key}`);
    const cardEl = document.getElementById(`mc-${m.key}`);

    if (valEl) valEl.textContent = Number.isInteger(val) ? val : val.toFixed(m.key==="error_rate"?1:1);
    if (barEl) {
      barEl.style.width       = pct + "%";
      barEl.style.background  =
        raw >= m.alertAt ? "var(--red)" :
        raw >= m.warnAt  ? "var(--amber)" : m.color;
    }
    if (cardEl) {
      cardEl.classList.toggle("alert", raw >= m.alertAt);
      cardEl.classList.toggle("warn",  raw >= m.warnAt && raw < m.alertAt);
    }

    // Update scenario label if anomalous
    if (data.scenario) {
      const sl = document.getElementById("scenario-label");
      sl.textContent = "⚠ " + data.scenario;
      sl.classList.add("visible");
    }
  });
}

// ── Chart ─────────────────────────────────────────────────────────────────────
function buildChart() {
  const ctx = document.getElementById("main-chart").getContext("2d");

  const META = {
    cpu_usage:    { label: "CPU %",        color: "#00d9ff" },
    memory_usage: { label: "Memory %",     color: "#a855f7" },
    latency:      { label: "Latency ÷ 10", color: "#f59e0b" },
    error_rate:   { label: "Error % × 10", color: "#ef4444" },
  };

  const dsets = CHART_METRICS.map(k => {
    datasets[k] = [];
    return {
      label: META[k].label,
      data:  datasets[k],
      borderColor: META[k].color,
      backgroundColor: META[k].color + "18",
      borderWidth: 1.8,
      pointRadius: 0,
      tension: 0.4,
      fill: false,
    };
  });

  chart = new Chart(ctx, {
    type: "line",
    data: { labels, datasets: dsets },
    options: {
      animation:   { duration: 300 },
      responsive:  true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          labels: { color: "#94a3b8", font: { size: 11 }, boxWidth: 12 }
        },
        tooltip: {
          backgroundColor: "rgba(8,12,20,0.92)",
          borderColor: "rgba(255,255,255,0.1)",
          borderWidth: 1,
          titleColor: "#e2e8f0",
          bodyColor:  "#94a3b8",
        }
      },
      scales: {
        x: {
          ticks: { color: "#334155", maxTicksLimit: 8, font:{ size:10 } },
          grid:  { color: "rgba(255,255,255,0.04)" },
        },
        y: {
          min: 0, max: 100,
          ticks: { color: "#334155", font:{ size:10 } },
          grid:  { color: "rgba(255,255,255,0.05)" },
        },
      },
    },
  });
}

function pushToChart(data) {
  const t = new Date().toLocaleTimeString("en-GB");
  labels.push(t);

  datasets.cpu_usage.push(round2(data.cpu_usage ?? 0));
  datasets.memory_usage.push(round2(data.memory_usage ?? 0));
  datasets.latency.push(round2((data.latency ?? 0) / 10));       // scale ÷10 to fit 0-100
  datasets.error_rate.push(round2((data.error_rate ?? 0) * 100 * 10)); // scale ×10

  if (labels.length > MAX_POINTS) {
    labels.shift();
    CHART_METRICS.forEach(k => datasets[k].shift());
  }

  chart.update("none");
}

// ── Stats Bar ─────────────────────────────────────────────────────────────────
function updateStats(stats) {
  document.getElementById("stat-heals").textContent     = stats.totalHeals     ?? 0;
  document.getElementById("stat-anomalies").textContent = stats.totalAnomalies ?? 0;
  const s = stats.uptimeSec ?? 0;
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  document.getElementById("stat-uptime").textContent =
    h ? `${h}h ${m}m` : m ? `${m}m ${sec}s` : `${sec}s`;
}

// ── Healing Log ───────────────────────────────────────────────────────────────
const TAG_MAP = {
  "CPU Spike":                 ["tag-cpu",   "CPU Spike"],
  "Memory Exhaustion (OOM)":   ["tag-mem",   "OOM"],
  "Disk I/O Saturation":       ["tag-disk",  "Disk I/O"],
  "Network Degradation":       ["tag-net",   "Network"],
  "Crash Loop":                ["tag-crash", "Crash Loop"],
};

function issueTag(issue) {
  const [cls, short] = TAG_MAP[issue] || ["tag-cpu", issue];
  return `<span class="tag ${cls}">${short}</span>`;
}

function addLogRow(ev) {
  const tbody = document.getElementById("log-body");
  // Remove "no events" placeholder
  const empty = tbody.querySelector(".log-empty");
  if (empty) empty.parentElement.remove();

  rowCounter++;
  const m = ev.metrics;
  const keyMetrics = m
    ? `CPU ${m.cpu_usage}%  MEM ${m.memory_usage}%  LAT ${m.latency}ms  ERR ${round2((m.error_rate||0)*100)}%`
    : "—";
  const ts = new Date(ev.time).toLocaleTimeString("en-GB") + " " +
             new Date(ev.time).toLocaleDateString("en-GB");

  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td style="color:var(--muted)">${rowCounter}</td>
    <td style="font-family:var(--mono);font-size:.75rem">${ts}</td>
    <td>${issueTag(ev.issue)} ${ev.issue}</td>
    <td style="font-family:var(--mono);font-size:.72rem;color:var(--muted)">${keyMetrics}</td>
    <td><span class="tag tag-action">${ev.action}</span></td>
  `;

  // Flash highlight
  tr.style.background = "rgba(34,197,94,0.06)";
  tbody.prepend(tr);
  setTimeout(() => tr.style.background = "", 3000);
}

// ── Demo Controls ─────────────────────────────────────────────────────────────
async function triggerStress() {
  await fetch("/stress");
  document.getElementById("scenario-label").textContent = "⚡ Stress mode active";
  document.getElementById("scenario-label").classList.add("visible");
}

async function resetSystem() {
  await fetch("/reset");
  document.getElementById("scenario-label").classList.remove("visible");
}
