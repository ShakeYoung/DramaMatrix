"""Minimal review web UI (U4).

一个零依赖（纯标准库）的本地审阅页面：渲染 review.json 的缩略图墙，
运营在浏览器里逐镜标记 approve / redraw / delete，保存时写入与
review_approver CLI 完全相同的 decisions.json——状态机应用
（apply_review_decisions）仍由重跑流水线统一执行，人工决定进状态机的
语义不被 UI 旁路。

用法：
    python -m src.review_server <project_id> <ep_key> [--port 8600] [--host 127.0.0.1]
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from src.agnes_video import output_root
from src.review import load_decisions, save_decisions

_PAGE = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>DramaMatrix 审阅台</title>
<style>
  body { font-family: -apple-system, "PingFang SC", sans-serif; margin: 24px; background: #f5f6f8; }
  h1 { font-size: 18px; }
  #bar { position: sticky; top: 0; background: #fff; padding: 10px 14px; border-radius: 8px;
         box-shadow: 0 1px 4px rgba(0,0,0,.12); display: flex; gap: 16px; align-items: center; }
  #grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 14px; margin-top: 16px; }
  .card { background: #fff; border-radius: 10px; padding: 10px; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
  .card img { width: 100%; border-radius: 6px; background: #ddd; min-height: 120px; object-fit: cover; }
  .meta { font-size: 12px; color: #555; margin: 6px 0; word-break: break-all; }
  .issues { color: #b00; font-size: 12px; }
  .opts { display: flex; gap: 8px; margin-top: 6px; }
  .opts label { flex: 1; text-align: center; border: 1px solid #ccc; border-radius: 6px; padding: 4px 0; font-size: 13px; cursor: pointer; }
  .opts input { display: none; }
  .opts input:checked + span { font-weight: 700; }
  .opts label:has(input:checked) { border-color: #16a; background: #eef6ff; }
  button { padding: 6px 18px; border: 0; border-radius: 6px; background: #1652a0; color: #fff; cursor: pointer; }
  #status { font-size: 13px; color: #333; }
</style>
</head>
<body>
<h1>DramaMatrix 逐镜人工审阅</h1>
<div id="bar">
  <span id="count">加载中…</span>
  <button onclick="save()">保存决定</button>
  <span id="status"></span>
</div>
<div id="grid"></div>
<script>
const params = new URLSearchParams(location.search);
const PROJECT = params.get('project') || '';
const EP = params.get('ep') || '';
let decisions = {};
async function load() {
  const [manifest, saved] = await Promise.all([
    fetch(`/api/manifest?project=${encodeURIComponent(PROJECT)}&ep=${encodeURIComponent(EP)}`).then(r => {
      if (!r.ok) throw new Error('审阅清单不存在（先跑流水线生成 review.json）');
      return r.json();
    }),
    fetch(`/api/decisions?project=${encodeURIComponent(PROJECT)}&ep=${encodeURIComponent(EP)}`).then(r => r.json()),
  ]);
  decisions = saved || {};
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  for (const shot of manifest.shots || []) {
    const card = document.createElement('div');
    card.className = 'card';
    const thumb = shot.thumbnail
      ? `<img src="/thumb?path=${encodeURIComponent(shot.thumbnail)}" loading="lazy">`
      : `<img src="data:image/gif;base64,R0lGODlhAQABAAAAACw=">`;
    const issues = (shot.qc_issues || []).map(t => `<div class="issues">⚠ ${t}</div>`).join('');
    card.innerHTML = `
      ${thumb}
      <div class="meta"><b>${shot.shot_id}</b>${shot.scene_id ? ' · ' + shot.scene_id : ''}
        ${shot.actual_duration ? ' · ' + Number(shot.actual_duration).toFixed(1) + 's' : ''}</div>
      <div class="meta">sha: ${(shot.sha256 || '').slice(0, 12)}</div>
      ${issues}
      <div class="opts">
        <label><input type="radio" name="${shot.shot_id}" value="approve"
          ${decisions[shot.shot_id] === 'approve' ? 'checked' : ''}><span>通过</span></label>
        <label><input type="radio" name="${shot.shot_id}" value="redraw"
          ${decisions[shot.shot_id] === 'redraw' ? 'checked' : ''}><span>重绘</span></label>
        <label><input type="radio" name="${shot.shot_id}" value="delete"
          ${decisions[shot.shot_id] === 'delete' ? 'checked' : ''}><span>删除</span></label>
      </div>`;
    grid.appendChild(card);
  }
  document.querySelectorAll('input[type=radio]').forEach(el => {
    el.addEventListener('change', () => { decisions[el.name] = el.value; updateCount(); });
  });
  updateCount();
}
function updateCount() {
  const total = document.querySelectorAll('.opts').length;
  const decided = Object.keys(decisions).filter(k => decisions[k]).length;
  document.getElementById('count').textContent = `已标记 ${decided}/${total} 镜`;
}
async function save() {
  const status = document.getElementById('status');
  status.textContent = '保存中…';
  const response = await fetch(
    `/api/decisions?project=${encodeURIComponent(PROJECT)}&ep=${encodeURIComponent(EP)}`,
    { method: 'POST', body: JSON.stringify(decisions) });
  status.textContent = response.ok ? '已保存。全部标记完成后，以同一 project_id --resume 续跑即可生效。' : '保存失败：' + await response.text();
}
load().catch(err => { document.getElementById('status').textContent = err.message; });
</script>
</body>
</html>
"""


class ReviewRequestHandler(BaseHTTPRequestHandler):
    server_version = "DramaMatrixReview/1.0"

    def _query(self) -> dict[str, str]:
        parsed = parse_qs(urlparse(self.path).query)
        return {key: values[0] for key, values in parsed.items()}

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        route = urlparse(self.path).path
        query = self._query()
        if route == "/" or route == "/index.html":
            self._send(200, _PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if route == "/api/manifest":
            path = _review_dir(query.get("project", ""), query.get("ep", "")) / "review.json"
            if not path.is_file():
                self._send_json(404, {"error": "review.json 不存在；请先让该集完成渲染生成审阅清单。"})
                return
            self._send(200, path.read_bytes(), "application/json; charset=utf-8")
            return
        if route == "/api/decisions":
            decisions = load_decisions(query.get("project", ""), query.get("ep", ""))
            self._send_json(200, decisions)
            return
        if route == "/thumb":
            self._serve_thumbnail(query.get("path", ""))
            return
        self._send_json(404, {"error": "unknown route"})

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        route = urlparse(self.path).path
        if route != "/api/decisions":
            self._send_json(404, {"error": "unknown route"})
            return
        query = self._query()
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            if not isinstance(payload, dict):
                raise ValueError("body must be an object")
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": f"无效的请求体：{exc}"})
            return
        path = save_decisions(query.get("project", ""), query.get("ep", ""), payload)
        self._send_json(200, {"saved": str(path), "count": len(payload)})

    def _serve_thumbnail(self, raw_path: str) -> None:
        # 只放行 outputs 根目录下的图片文件，杜绝路径穿越。
        try:
            candidate = Path(unquote(raw_path)).resolve()
            candidate.relative_to(output_root().resolve())
        except (ValueError, OSError):
            self._send_json(403, {"error": "路径不在输出目录内"})
            return
        if candidate.suffix.lower() not in {".png", ".jpg", ".jpeg"} or not candidate.is_file():
            self._send_json(404, {"error": "缩略图不存在"})
            return
        self._send(200, candidate.read_bytes(), "image/png")

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - stdlib signature
        print(f"[review-server] {self.address_string()} {format % args}")


def _review_dir(project_id: str, ep_key: str) -> Path:
    from src.review import _review_dir as base

    return base(project_id, ep_key)


def serve(project_id: str, ep_key: str, host: str = "127.0.0.1", port: int = 8600) -> None:
    handler = ReviewRequestHandler
    httpd = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/?project={project_id}&ep={ep_key}"
    print(f"👤 审阅台已启动：{url}")
    print("   浏览器中标记并保存后，关闭本服务并以同一 project_id --resume 续跑生效。")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n审阅台已停止。")
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print("用法：python -m src.review_server <project_id> <ep_key> [--port 8600] [--host 127.0.0.1]")
        return 2
    project_id, ep_key = argv[0], argv[1]
    host, port = "127.0.0.1", 8600
    index = 2
    while index < len(argv) - 1:
        if argv[index] == "--port":
            port = int(argv[index + 1])
            index += 2
        elif argv[index] == "--host":
            host = argv[index + 1]
            index += 2
        else:
            index += 1
    serve(project_id, ep_key, host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
