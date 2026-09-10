"""Operations dashboard (W5) —— 只读运营看板。

直接聚合 SQLite 证据链（快照/用量/QC/评分/状态历史），零依赖标准库实现：

    python -m src.dashboard_server [--port 8700] [--host 127.0.0.1]

页面：
- 项目列表：系统状态、各集状态分布、创建次数（成本代理）。
- 项目详情：逐集进度、供应商用量（排队/渲染/下载耗时、重绘次数）、
  QC 通过率与指标均值（亮度差/结构相似度/身份分）、人工评分均值、
  最近状态版本（state_history）。

只读——不写任何表；所有数字来自生产过程沉淀的证据，而非另行上报。
"""

from __future__ import annotations

import json
import sqlite3
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

_PAGE = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>DramaMatrix 运营看板</title>
<style>
  body { font-family: -apple-system, "PingFang SC", sans-serif; margin: 24px; background: #f5f6f8; }
  h1 { font-size: 20px; } h2 { font-size: 15px; margin: 18px 0 8px; }
  table { border-collapse: collapse; background: #fff; width: 100%; border-radius: 8px; overflow: hidden;
          box-shadow: 0 1px 3px rgba(0,0,0,.1); }
  th, td { padding: 7px 10px; font-size: 13px; text-align: left; border-bottom: 1px solid #eee; }
  th { background: #1652a0; color: #fff; font-weight: 500; }
  .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 10px; margin: 10px 0; }
  .kpi { background: #fff; border-radius: 8px; padding: 10px 12px; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
  .kpi b { display: block; font-size: 20px; }
  .kpi span { font-size: 12px; color: #666; }
  .pill { display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 12px; background: #eef; }
  .project-row { cursor: pointer; }
  .project-row:hover { background: #f0f6ff; }
  .active { background: #eef6ff !important; }
  #detail { margin-top: 12px; }
</style>
</head>
<body>
<h1>DramaMatrix 运营看板（只读）</h1>
<div class="cards" id="kpis"></div>
<h2>项目</h2>
<table id="projects"><thead><tr>
  <th>项目</th><th>系统状态</th><th>集数分布</th><th>创建任务</th><th>重绘次数</th><th>更新时间</th>
</tr></thead><tbody></tbody></table>
<div id="detail"></div>
<script>
let current = null;
async function loadProjects() {
  const data = await (await fetch('/api/projects')).json();
  const kpis = document.getElementById('kpis');
  kpis.innerHTML = [
    ['项目数', data.summary.projects],
    ['创建任务', data.summary.creates],
    ['重绘次数', data.summary.redraws],
    ['QC 通过率', data.summary.qc_pass_rate],
    ['平均结构相似度', data.summary.avg_similarity],
  ].map(([k, v]) => `<div class="kpi"><b>${v ?? '—'}</b><span>${k}</span></div>`).join('');
  const tbody = document.querySelector('#projects tbody');
  tbody.innerHTML = '';
  for (const p of data.projects) {
    const tr = document.createElement('tr');
    tr.className = 'project-row';
    tr.innerHTML = `<td>${p.project_id}</td><td>${p.system_status}</td>
      <td>${p.status_brief}</td><td>${p.creates}</td><td>${p.redraws}</td><td>${p.updated_at || ''}</td>`;
    tr.onclick = () => { current = p.project_id; loadProject(current);
      document.querySelectorAll('.project-row').forEach(r => r.classList.remove('active'));
      tr.classList.add('active'); };
    tbody.appendChild(tr);
  }
}
async function loadProject(id) {
  const p = await (await fetch(`/api/project/${encodeURIComponent(id)}`)).json();
  const detail = document.getElementById('detail');
  const eps = p.episodes.map(e =>
    `<tr><td>${e.ep_key}</td><td>${e.status}</td><td>${e.rendered}/${e.planned}</td>
     <td>${e.storyboard_version}</td></tr>`).join('');
  const provs = p.providers.map(r =>
    `<tr><td>${r.provider || '—'}</td><td>${r.creates}</td><td>${r.redraws}</td>
     <td>${r.avg_queue ?? '—'}</td><td>${r.avg_render ?? '—'}</td><td>${r.avg_download ?? '—'}</td></tr>`).join('');
  const qc = p.qc.map(r =>
    `<tr><td>${r.ep_key}</td><td>${r.total}</td><td>${r.passed}</td><td>${r.pass_rate}</td>
     <td>${r.avg_brightness ?? '—'}</td><td>${r.avg_similarity ?? '—'}</td><td>${r.avg_identity ?? '—'}</td></tr>`).join('');
  detail.innerHTML = `
    <h2>剧集进度 · ${id}</h2>
    <table><thead><tr><th>集</th><th>状态</th><th>已渲染/计划</th><th>分镜版本</th></tr></thead><tbody>${eps}</tbody></table>
    <h2>供应商用量</h2>
    <table><thead><tr><th>供应商</th><th>创建</th><th>重绘</th><th>平均排队(s)</th><th>平均渲染(s)</th><th>平均下载(s)</th></tr></thead><tbody>${provs}</tbody></table>
    <h2>QC 统计</h2>
    <table><thead><tr><th>集</th><th>样本</th><th>通过</th><th>通过率</th><th>平均亮度差</th><th>平均相似度</th><th>平均身份分</th></tr></thead><tbody>${qc}</tbody></table>`;
}
loadProjects();
</script>
</body>
</html>
"""


def _connect() -> sqlite3.Connection:
    import src.db as db_module

    conn = sqlite3.connect(db_module.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _fmt(value, digits=2):
    return None if value is None else round(value, digits)


def projects_payload() -> dict:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT project_id, system_status, state_json, updated_at FROM project_state_snapshots ORDER BY updated_at DESC"
        ).fetchall()
        usage = {
            row["project_id"]: dict(row)
            for row in conn.execute(
                "SELECT project_id, COUNT(*) AS creates, COALESCE(SUM(redraw_count), 0) AS redraws"
                " FROM agnes_usage GROUP BY project_id"
            )
        }
        qc_all = {
            row[0]: (row[1], row[2])
            for row in conn.execute(
                "SELECT project_id, COUNT(*), SUM(passed) FROM shot_qc_results GROUP BY project_id"
            )
        }
    finally:
        conn.close()

    projects = []
    for row in rows:
        try:
            episodes = (json.loads(row["state_json"]) or {}).get("episodes") or {}
        except json.JSONDecodeError:
            episodes = {}
        counts: dict[str, int] = {}
        for ep in episodes.values():
            status = ep.get("status", "?") if isinstance(ep, dict) else getattr(ep, "status", "?")
            counts[status] = counts.get(status, 0) + 1
        brief = "；".join(f"{k}×{v}" for k, v in sorted(counts.items())) or "—"
        stat = usage.get(row["project_id"], {})
        qc_total, qc_passed = qc_all.get(row["project_id"], (0, 0))
        projects.append({
            "project_id": row["project_id"],
            "system_status": row["system_status"],
            "episode_count": len(episodes),
            "status_brief": brief,
            "creates": stat.get("creates", 0) if stat else 0,
            "redraws": stat.get("redraws", 0) if stat else 0,
            "qc_total": qc_total,
            "qc_passed": qc_passed,
            "updated_at": row["updated_at"],
        })

    total_creates = sum(p["creates"] for p in projects)
    total_redraws = sum(p["redraws"] for p in projects)
    total_qc = sum(p["qc_total"] for p in projects)
    passed_qc = sum(p["qc_passed"] for p in projects)
    conn = _connect()
    try:
        sim_row = conn.execute(
            "SELECT AVG(CAST(json_extract(metrics, '$.frame_similarity') AS REAL)) FROM shot_qc_results"
        ).fetchone()
        avg_similarity = _fmt(sim_row[0], 3) if sim_row and sim_row[0] is not None else None
    except sqlite3.Error:
        avg_similarity = None
    finally:
        conn.close()
    return {
        "projects": projects,
        "summary": {
            "projects": len(projects),
            "creates": total_creates,
            "redraws": total_redraws,
            "qc_pass_rate": f"{passed_qc / total_qc * 100:.0f}%" if total_qc else None,
            "avg_similarity": avg_similarity,
        },
    }


def project_payload(project_id: str) -> dict:
    conn = _connect()
    try:
        snapshot = conn.execute(
            "SELECT state_json FROM project_state_snapshots WHERE project_id = ?", (project_id,)
        ).fetchone()
        providers = [
            {
                "provider": row["provider"],
                "creates": row["creates"],
                "redraws": row["redraws"],
                "avg_queue": _fmt(row["avg_queue"]),
                "avg_render": _fmt(row["avg_render"]),
                "avg_download": _fmt(row["avg_download"]),
            }
            for row in conn.execute(
                """SELECT provider, COUNT(*) AS creates, COALESCE(SUM(redraw_count), 0) AS redraws,
                          AVG(queue_wait_seconds) AS avg_queue, AVG(render_seconds) AS avg_render,
                          AVG(download_seconds) AS avg_download
                   FROM agnes_usage WHERE project_id = ? GROUP BY provider""",
                (project_id,),
            )
        ]
        qc_rows = conn.execute(
            """SELECT ep_key, COUNT(*) AS total, SUM(passed) AS passed,
                      AVG(brightness_diff) AS avg_brightness,
                      AVG(CAST(json_extract(metrics, '$.frame_similarity') AS REAL)) AS avg_similarity,
                      AVG(CAST(json_extract(metrics, '$.identity_score') AS REAL)) AS avg_identity
               FROM shot_qc_results WHERE project_id = ? GROUP BY ep_key""",
            (project_id,),
        ).fetchall()
        qc = [
            {
                "ep_key": row["ep_key"] or "—",
                "total": row["total"],
                "passed": row["passed"],
                "pass_rate": f"{row['passed'] / row['total'] * 100:.0f}%" if row["total"] else "—",
                "avg_brightness": _fmt(row["avg_brightness"]),
                "avg_similarity": _fmt(row["avg_similarity"], 3),
                "avg_identity": _fmt(row["avg_identity"]),
            }
            for row in qc_rows
        ]
        versions = [
            dict(row)
            for row in conn.execute(
                "SELECT version, system_status, recorded_at FROM state_history"
                " WHERE project_id = ? ORDER BY version DESC LIMIT 10",
                (project_id,),
            )
        ]
    finally:
        conn.close()

    episodes = []
    if snapshot:
        try:
            eps = (json.loads(snapshot["state_json"]) or {}).get("episodes") or {}
        except json.JSONDecodeError:
            eps = {}
        for key, ep in eps.items():
            data = ep if isinstance(ep, dict) else ep.model_dump()
            episodes.append({
                "ep_key": key,
                "status": data.get("status", "?"),
                "rendered": data.get("rendered_shot_count", 0),
                "planned": data.get("planned_shot_count", len(data.get("storyboard_data") or [])),
                "storyboard_version": data.get("storyboard_version", 1),
            })
    return {"project_id": project_id, "episodes": episodes, "providers": providers, "qc": qc,
            "state_versions": versions}


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "DramaMatrixDashboard/1.0"

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._send(200, _PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/projects":
            try:
                self._json(200, projects_payload())
            except sqlite3.Error as exc:
                self._json(500, {"error": str(exc)})
            return
        if parsed.path.startswith("/api/project/"):
            project_id = parsed.path[len("/api/project/"):].strip("/")
            if not project_id:
                self._json(400, {"error": "missing project id"})
                return
            try:
                self._json(200, project_payload(project_id))
            except sqlite3.Error as exc:
                self._json(500, {"error": str(exc)})
            return
        self._json(404, {"error": "unknown route"})

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - stdlib signature
        print(f"[dashboard] {self.address_string()} {format % args}")


def serve(host: str = "127.0.0.1", port: int = 8700) -> None:
    httpd = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"📊 运营看板已启动：http://{host}:{port}/ （只读，Ctrl+C 停止）")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n看板已停止。")
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    host, port = "127.0.0.1", 8700
    index = 0
    while index < len(argv) - 1:
        if argv[index] == "--port":
            port = int(argv[index + 1])
            index += 2
        elif argv[index] == "--host":
            host = argv[index + 1]
            index += 2
        else:
            index += 1
    serve(host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
