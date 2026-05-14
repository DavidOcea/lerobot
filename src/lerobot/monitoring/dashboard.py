"""
Real-time monitoring dashboard for robot + AGV state visualization.

Architecture:
  MonitorCollector (independent thread)
    ├── Reads robot state from lock-protected shared dict (updated by main loop)
    ├── Polls AGV batch APIs (1100 + 1102) at 500ms intervals
    └── Exposes aggregated state via thread-safe .snapshot property

  HTTPDashboard (stdlib http.server, separate thread)
    ├── GET /api/status → JSON snapshot
    └── GET / → dashboard.html (single-page SPA)

The main control loop only does one non-blocking write per frame:
    monitor_collector.update_robot_state(observation, action, task_info)
This is lock-protected and will never block the 50Hz control loop.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RobotSnapshot:
    """Latest robot joint state snapshot."""
    positions: dict[str, float] = field(default_factory=dict)  # {joint_name: degrees}
    forces: dict[str, float] = field(default_factory=dict)     # {joint_name: Nm}
    fps: float = 0.0
    timestamp: float = 0.0


@dataclass
class AGVSnapshot:
    """Latest AGV state snapshot from batch APIs."""
    connected: bool = False
    battery_pct: int = 0          # 0-100
    battery_voltage: float = 0.0  # V
    charging: bool = False
    x: float = 0.0                # m
    y: float = 0.0                # m
    theta: float = 0.0            # rad (heading)
    vx: float = 0.0               # m/s
    vy: float = 0.0               # m/s
    vtheta: float = 0.0           # rad/s
    current_station: str = ""
    running_status: int = 0       # 0=idle, 1=executing, 2=charging, 3=error, 4=paused
    emergency: bool = False
    is_moving: bool = False
    errors: list[str] = field(default_factory=list)
    odo: float = 0.0              # cumulative distance (m)
    controller_temp: float = 0.0  # °C
    timestamp: float = 0.0


@dataclass
class TaskSnapshot:
    """Latest task execution snapshot."""
    task_name: str = ""
    task_type: str = ""
    cycle: int = 0
    total_cycles: int = 0
    collision_count: int = 0
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    last_error: str = ""
    timestamp: float = 0.0


class MonitorCollector:
    """Thread-safe collector that aggregates robot + AGV state.

    The main control loop calls update_robot_state() each frame
    (non-blocking, lock-protected).  A background thread polls the
    AGV batch APIs independently.
    """

    def __init__(self, agv_controller=None, robot=None):
        self._agv = agv_controller
        self._robot = robot
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

        self._robot_snapshot = RobotSnapshot()
        self._agv_snapshot = AGVSnapshot()
        self._task_snapshot = TaskSnapshot()

        # FPS tracking
        self._frame_count = 0
        self._fps_window_start = time.time()
        self._current_fps = 0.0

        # AGV poll interval
        self._agv_poll_interval = 0.5  # 500ms
        self._last_agv_poll = 0.0

        # Error/warning ring buffer
        self._event_log: list[dict] = []  # [{ts, level, source, message}]
        self._max_events = 200

    # ── Public API (called from main loop, non-blocking) ──────────

    def update_robot_state(
        self,
        observation: dict | None = None,
        action: dict | None = None,
        task_info: dict | None = None,
    ):
        """Called from main loop every frame.  Non-blocking.

        Args:
            observation: Raw observation dict from robot.get_observation()
            action: Sent action dict (optional)
            task_info: {"task_name", "task_type", "cycle", "total_cycles", ...}
        """
        # Extract positions and forces from observation
        positions = {}
        forces = {}
        if observation:
            for key, value in observation.items():
                if key.endswith(".pos"):
                    positions[key.removesuffix(".pos")] = float(value)
                elif key.endswith(".force"):
                    forces[key.removesuffix(".force")] = float(value)

        # FPS tracking
        self._frame_count += 1
        now = time.time()
        elapsed = now - self._fps_window_start
        if elapsed >= 1.0:
            self._current_fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_window_start = now

        with self._lock:
            self._robot_snapshot = RobotSnapshot(
                positions=positions,
                forces=forces,
                fps=self._current_fps,
                timestamp=now,
            )
            if task_info:
                self._task_snapshot = TaskSnapshot(
                    task_name=task_info.get("task_name", ""),
                    task_type=task_info.get("task_type", ""),
                    cycle=task_info.get("cycle", 0),
                    total_cycles=task_info.get("total_cycles", 0),
                    collision_count=task_info.get("collision_count", 0),
                    total_tasks=task_info.get("total_tasks", 0),
                    completed_tasks=task_info.get("completed_tasks", 0),
                    failed_tasks=task_info.get("failed_tasks", 0),
                    last_error=task_info.get("last_error", ""),
                    timestamp=now,
                )

    def add_event(self, level: str, source: str, message: str):
        """Log a monitoring event (thread-safe)."""
        with self._lock:
            self._event_log.append({
                "ts": time.time(),
                "level": level,
                "source": source,
                "message": message,
            })
            if len(self._event_log) > self._max_events:
                self._event_log = self._event_log[-self._max_events:]

    # ── Background thread ────────────────────────────────────────

    def start(self):
        """Start the background AGV polling thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="monitor-collector")
        self._thread.start()
        logger.info("MonitorCollector started (background thread)")

    def stop(self):
        """Stop the background thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("MonitorCollector stopped")

    def _poll_loop(self):
        """Background thread: poll AGV state at fixed interval."""
        while self._running:
            try:
                now = time.time()
                if now - self._last_agv_poll >= self._agv_poll_interval:
                    self._poll_agv()
                    self._last_agv_poll = now
            except Exception as e:
                logger.error(f"MonitorCollector poll error: {e}")
            time.sleep(0.1)  # 100ms tick

    def _poll_agv(self):
        """Poll AGV batch APIs and update snapshot."""
        if self._agv is None:
            with self._lock:
                self._agv_snapshot = AGVSnapshot(connected=False)
            return

        try:
            if not self._agv.is_connected():
                with self._lock:
                    self._agv_snapshot = AGVSnapshot(connected=False)
                return

            # Use existing get_status() which calls the 1100 aggregate API
            status = self._agv.get_status(use_cache=False)

            # Get detailed battery from 1102 API
            battery_voltage = 0.0
            charging = False
            controller_temp = 0.0
            odo = 0.0
            alarms: list[str] = []

            try:
                batt_resp = self._agv._send_request(
                    self._agv.PORT_STATUS,
                    0x044E,
                    {}
                )
                battery_voltage = float(batt_resp.get("controller_voltage", 0.0))
                charging = batt_resp.get("charging", False)
                controller_temp = float(batt_resp.get("controller_temp", 0.0))
            except Exception:
                pass

            try:
                odo_resp = self._agv._send_request(
                    self._agv.PORT_STATUS,
                    0x03EA,
                    {}
                )
                odo = float(odo_resp.get("odo", 0.0))
            except Exception:
                pass

            try:
                err_resp = self._agv._send_request(
                    self._agv.PORT_STATUS,
                    0x041A,
                    {}
                )
                alarms = [str(a) for a in err_resp.get("errors", [])]
                fatals_list = [str(a) for a in err_resp.get("fatals", [])]
                alarms.extend(fatals_list)
            except Exception:
                pass

            with self._lock:
                self._agv_snapshot = AGVSnapshot(
                    connected=True,
                    battery_pct=status.battery,
                    battery_voltage=battery_voltage,
                    charging=charging,
                    x=status.position.x,
                    y=status.position.y,
                    theta=status.position.theta,
                    vx=status.vx,
                    vy=status.vy,
                    vtheta=status.vtheta,
                    current_station=status.current_station,
                    running_status=status.status_code,
                    emergency=status.error_code != 0,
                    is_moving=status.is_moving,
                    errors=alarms,
                    odo=odo,
                    controller_temp=controller_temp,
                    timestamp=time.time(),
                )

        except Exception as e:
            logger.error(f"AGV poll failed: {e}")
            with self._lock:
                self._agv_snapshot = AGVSnapshot(connected=False)

    # ── Thread-safe snapshot access ──────────────────────────────

    def get_full_status(self) -> dict:
        """Return complete monitoring status as a JSON-serializable dict."""
        with self._lock:
            robot = self._robot_snapshot
            agv = self._agv_snapshot
            task = self._task_snapshot
            events = list(self._event_log[-50:])

        return {
            "timestamp": time.time(),
            "robot": {
                "positions": robot.positions,
                "forces": robot.forces,
                "fps": round(robot.fps, 1),
                "connected": len(robot.positions) > 0,
                "joint_count": len(robot.positions),
                "joint_names": list(robot.positions.keys()),
                "last_update": robot.timestamp,
            },
            "agv": {
                "connected": agv.connected,
                "battery_pct": agv.battery_pct,
                "battery_voltage": round(agv.battery_voltage, 1),
                "charging": agv.charging,
                "position": {
                    "x": round(agv.x, 3),
                    "y": round(agv.y, 3),
                    "theta": round(agv.theta, 3),
                    "theta_deg": round(agv.theta * 57.2958, 1),
                },
                "velocity": {
                    "vx": round(agv.vx, 3),
                    "vy": round(agv.vy, 3),
                    "vtheta": round(agv.vtheta, 3),
                },
                "current_station": agv.current_station,
                "running_status": agv.running_status,
                "running_status_label": _status_label(agv.running_status),
                "emergency": agv.emergency,
                "is_moving": agv.is_moving,
                "errors": agv.errors,
                "odo": round(agv.odo, 1),
                "controller_temp": round(agv.controller_temp, 1),
                "last_update": agv.timestamp,
            },
            "task": {
                "task_name": task.task_name,
                "task_type": task.task_type,
                "cycle": task.cycle,
                "total_cycles": task.total_cycles,
                "collision_count": task.collision_count,
                "total_tasks": task.total_tasks,
                "completed_tasks": task.completed_tasks,
                "failed_tasks": task.failed_tasks,
                "last_error": task.last_error,
                "last_update": task.timestamp,
            },
            "events": events,
        }


def _status_label(code: int) -> str:
    labels = {0: "IDLE", 1: "EXECUTING", 2: "CHARGING", 3: "ERROR", 4: "PAUSED"}
    return labels.get(code, f"UNKNOWN({code})")


# ═══════════════════════════════════════════════════════════════════
# HTTP Dashboard Server (stdlib http.server, zero dependencies)
# ═══════════════════════════════════════════════════════════════════

_DASHBOARD_HTML = None


def _get_dashboard_html() -> str:
    """Load dashboard HTML from file or use embedded minimal version."""
    global _DASHBOARD_HTML
    if _DASHBOARD_HTML is not None:
        return _DASHBOARD_HTML

    html_path = Path(__file__).parent / "dashboard.html"
    if html_path.exists():
        _DASHBOARD_HTML = html_path.read_text()
        return _DASHBOARD_HTML

    _DASHBOARD_HTML = _MINIMAL_DASHBOARD
    return _DASHBOARD_HTML


class _DashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the monitoring dashboard."""

    collector: "MonitorCollector | None" = None

    def log_message(self, format, *args):
        logger.debug(f"Dashboard HTTP: {format % args}")

    def do_GET(self):
        if self.path == "/api/status":
            self._serve_json()
        elif self.path in ("/", "/index.html"):
            self._serve_html()
        elif self.path == "/health":
            self._serve_json({"status": "ok"})
        else:
            self.send_error(404)

    def _serve_json(self, data=None):
        if data is None and self.collector:
            data = self.collector.get_full_status()
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _serve_html(self):
        html = _get_dashboard_html()
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)


class HTTPDashboard:
    """Lightweight HTTP server for the monitoring dashboard.

    Runs in its own daemon thread.  Serves:
    - /api/status  — JSON snapshot from MonitorCollector
    - /            — Single-page HTML dashboard

    Usage:
        dashboard = HTTPDashboard(collector, port=8080)
        dashboard.start()
    """

    def __init__(self, collector: MonitorCollector, port: int = 8080):
        self.collector = collector
        self.port = port
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self):
        """Start HTTP server in background thread."""
        _DashboardHandler.collector = self.collector

        self._server = HTTPServer(("0.0.0.0", self.port), _DashboardHandler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="monitor-dashboard",
        )
        self._thread.start()
        logger.info(f"Monitoring dashboard: http://0.0.0.0:{self.port}")

    def stop(self):
        """Stop HTTP server."""
        if self._server:
            self._server.shutdown()
        if self._thread:
            self._thread.join(timeout=2.0)


_MINIMAL_DASHBOARD = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Robot Monitor</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:monospace;background:#0a0a0f;color:#c0c0c0;padding:20px}
h1{color:#00ff88;margin-bottom:16px}
.status-bar{display:flex;gap:16px;margin-bottom:16px;flex-wrap:wrap}
.card{background:#141420;border:1px solid #2a2a3a;border-radius:8px;padding:14px;min-width:200px;flex:1}
.card h3{color:#888;font-size:12px;text-transform:uppercase;margin-bottom:6px}
.card .value{font-size:22px;font-weight:bold}
.ok{color:#00ff88}.warn{color:#ffaa00}.err{color:#ff4444}
.joint-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:10px;margin-bottom:16px}
.joint-row{display:flex;align-items:center;gap:8px;font-size:12px}
.joint-name{width:140px;text-align:right;color:#888}
.joint-bar{flex:1;height:14px;background:#1a1a2a;border-radius:4px;overflow:hidden;position:relative}
.joint-fill{height:100%;border-radius:4px;transition:width .3s}
.joint-val{width:60px;text-align:right}
.l-arm .joint-fill{background:#4488ff} .r-arm .joint-fill{background:#ff8844}
.trunk .joint-fill{background:#aa44ff}
.event-log{margin-top:16px}
.event-log h3{color:#888;font-size:12px;text-transform:uppercase;margin-bottom:8px}
.event-table{width:100%;font-size:11px;border-collapse:collapse;table-layout:fixed}
.event-table col.col-time{width:90px}
.event-table col.col-level{width:52px}
.event-table col.col-source{width:155px}
.event-table th{color:#666;text-align:left;padding:4px 8px;border-bottom:1px solid #2a2a3a}
.event-table td{padding:3px 8px;border-bottom:1px solid #1a1a2a;vertical-align:top}
.event-table .ev-time{color:#555}
.event-table .ev-level{}
.event-table .ev-source{color:#888}
.event-table .ev-source span{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.event-table .ev-msg{color:#c0c0c0;word-break:break-word;overflow-wrap:break-word}
.event-table tr.ev-warn{background:#332200}
.event-table tr.ev-error{background:#330000}
pre{font-size:11px;color:#666;max-height:200px;overflow-y:auto;margin-top:16px}
</style>
</head>
<body>
<h1>Robot Monitoring</h1>
<div id="status"></div>
<pre id="raw"></pre>
<script>
async function refresh(){
  try{
    const r=await fetch('/api/status');
    const d=await r.json();
    document.getElementById('raw').textContent=JSON.stringify(d,null,2);
    document.getElementById('status').textContent='';
    document.getElementById('status').appendChild(buildUI(d));
  }catch(e){}
  setTimeout(refresh,1000);
}
function buildUI(d){
  const frag=document.createDocumentFragment();
  const r=d.robot||{},a=d.agv||{},t=d.task||{};
  const bar=document.createElement('div'); bar.className='status-bar';
  bar.appendChild(card('Robot FPS',(r.fps||0).toFixed(1),r.connected?'ok':'err'));
  bar.appendChild(card('Battery',(a.battery_pct||0)+'%',a.battery_pct>20?'ok':(a.battery_pct>10?'warn':'err')));
  bar.appendChild(card('AGV Status',a.running_status_label||'N/A',a.connected?'ok':'err'));
  bar.appendChild(card('Emergency',a.emergency?'ACTIVE':'OK',a.emergency?'err':'ok'));
  bar.appendChild(card('Task',t.task_name||'-',t.task_type||''));
  bar.appendChild(card('Cycle',t.cycle+'/'+t.total_cycles,''));
  frag.appendChild(bar);
  if(Object.keys(r.positions||{}).length){
    const grid=document.createElement('div'); grid.className='joint-grid';
    const names=['left_arm','right_arm','trunk'];
    for(const n of names){
      for(const[k,v]of Object.entries(r.positions||{})){
        if(!k.startsWith(n))continue;
        const cls=k.startsWith('left')?'l-arm':k.startsWith('right')?'r-arm':'trunk';
        const pct=Math.min(100,Math.abs(v)/180*100);
        const row=document.createElement('div'); row.className='joint-row '+cls;
        const nm=document.createElement('span'); nm.className='joint-name'; nm.textContent=k;
        const barDiv=document.createElement('div'); barDiv.className='joint-bar';
        const fill=document.createElement('div'); fill.className='joint-fill'; fill.style.width=pct+'%';
        barDiv.appendChild(fill);
        const val=document.createElement('span'); val.className='joint-val'; val.textContent=v.toFixed(1)+'°';
        row.appendChild(nm); row.appendChild(barDiv); row.appendChild(val);
        grid.appendChild(row);
      }
    }
    frag.appendChild(grid);
  }
  if((d.events||[]).length){
    const evDiv=document.createElement('div'); evDiv.className='event-log';
    const evHdr=document.createElement('h3'); evHdr.textContent='Event Log';
    evDiv.appendChild(evHdr);
    const tbl=document.createElement('table'); tbl.className='event-table';
    const colg=document.createElement('colgroup');
    colg.innerHTML='<col class="col-time"><col class="col-level"><col class="col-source"><col>';
    tbl.appendChild(colg);
    const thead=document.createElement('thead');
    thead.innerHTML='<tr><th>Time</th><th>Lvl</th><th>Source</th><th>Message</th></tr>';
    tbl.appendChild(thead);
    const tbody=document.createElement('tbody');
    for(const e of d.events.slice(-30)){
      const row=document.createElement('tr');
      if(e.level==='warn') row.className='ev-warn';
      else if(e.level==='error') row.className='ev-error';
      const ts=new Date(e.ts*1000).toLocaleTimeString();
      row.innerHTML='<td class="ev-time">'+ts+'</td>'
        +'<td class="ev-level">'+e.level+'</td>'
        +'<td class="ev-source"><span>'+e.source+'</span></td>'
        +'<td class="ev-msg">'+e.message+'</td>';
      tbody.appendChild(row);
    }
    tbl.appendChild(tbody);
    evDiv.appendChild(tbl);
    frag.appendChild(evDiv);
  }
  return frag;
}
function card(title,val,cls){
  const c=document.createElement('div'); c.className='card';
  const h=document.createElement('h3'); h.textContent=title;
  const v=document.createElement('div'); v.className='value '+(cls||''); v.textContent=val;
  c.appendChild(h); c.appendChild(v);
  return c;
}
refresh();
</script>
</body>
</html>"""
