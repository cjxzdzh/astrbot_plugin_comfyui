# -*- coding: utf-8 -*-
"""
工作流管理小站：提供 .json 文件列表、上传、备注、重命名、删除。
在浏览器打开 http://localhost:{management_port} 使用。
"""
import re
from pathlib import Path
from typing import Callable, Dict

from aiohttp import web

# 安全文件名：只保留安全字符
SAFE_FILENAME_RE = re.compile(r"^[a-zA-Z0-9_\-\+\.\u4e00-\u9fff]+$")


def _safe_basename(name: str) -> str:
    """防止路径穿越，只取 basename。"""
    return Path(name).name.strip()


def create_app(
    workflows_dir: Path,
    meta_path: Path,
    load_meta: Callable[[], Dict[str, str]],
    save_meta: Callable[[Dict[str, str]], None],
) -> web.Application:
    app = web.Application()

    async def list_handler(_: web.Request) -> web.Response:
        """GET /api/list：列出 workflows 目录下所有 .json 及备注。"""
        meta = load_meta()
        files = []
        if workflows_dir.exists():
            for f in sorted(workflows_dir.glob("*.json")):
                name = f.name
                files.append({
                    "filename": name,
                    "description": meta.get(name, ""),
                })
        return web.json_response({"files": files})

    async def upload_handler(request: web.Request) -> web.Response:
        """POST /api/upload：上传一个 .json 文件。"""
        reader = await request.multipart()
        field = None
        while True:
            part = await reader.next()
            if part is None:
                break
            if part.name == "file":
                field = part
                break
        if field is None:
            return web.json_response({"ok": False, "error": "missing field: file"}, status=400)
        filename = _safe_basename(field.filename or "workflow.json")
        if not filename.lower().endswith(".json"):
            return web.json_response({"ok": False, "error": "only .json allowed"}, status=400)
        if not SAFE_FILENAME_RE.match(filename):
            return web.json_response({"ok": False, "error": "invalid filename"}, status=400)
        workflows_dir.mkdir(parents=True, exist_ok=True)
        path = workflows_dir / filename
        size = 0
        with open(path, "wb") as out:
            while True:
                chunk = await field.read_chunk()
                if not chunk:
                    break
                size += len(chunk)
                out.write(chunk)
        return web.json_response({"ok": True, "filename": filename, "size": size})

    async def description_handler(request: web.Request) -> web.Response:
        """POST /api/description：保存某个文件的备注。body: {"filename":"x.json","description":"..."}"""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid json"}, status=400)
        filename = _safe_basename(body.get("filename") or "")
        description = (body.get("description") or "").strip()
        if not filename or not filename.endswith(".json"):
            return web.json_response({"ok": False, "error": "invalid filename"}, status=400)
        meta = load_meta()
        meta[filename] = description
        save_meta(meta)
        return web.json_response({"ok": True})

    async def rename_handler(request: web.Request) -> web.Response:
        """POST /api/rename：重命名文件。body: {"old_name":"a.json","new_name":"b.json"}"""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid json"}, status=400)
        old_name = _safe_basename(body.get("old_name") or "")
        new_name = _safe_basename(body.get("new_name") or "")
        if not old_name.endswith(".json") or not new_name.endswith(".json"):
            return web.json_response({"ok": False, "error": "only .json allowed"}, status=400)
        if not SAFE_FILENAME_RE.match(new_name):
            return web.json_response({"ok": False, "error": "invalid new filename"}, status=400)
        old_path = workflows_dir / old_name
        new_path = workflows_dir / new_name
        if not old_path.exists():
            return web.json_response({"ok": False, "error": "file not found"}, status=404)
        if new_path.exists():
            return web.json_response({"ok": False, "error": "target already exists"}, status=400)
        old_path.rename(new_path)
        meta = load_meta()
        if old_name in meta:
            meta[new_name] = meta.pop(old_name)
            save_meta(meta)
        return web.json_response({"ok": True, "filename": new_name})

    async def delete_handler(request: web.Request) -> web.Response:
        """POST /api/delete：删除文件。body: {"filename":"x.json"}"""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid json"}, status=400)
        filename = _safe_basename(body.get("filename") or "")
        if not filename.endswith(".json"):
            return web.json_response({"ok": False, "error": "invalid filename"}, status=400)
        path = workflows_dir / filename
        if not path.exists():
            return web.json_response({"ok": False, "error": "file not found"}, status=404)
        path.unlink()
        meta = load_meta()
        meta.pop(filename, None)
        save_meta(meta)
        return web.json_response({"ok": True})

    async def index_handler(_: web.Request) -> web.Response:
        """GET /：返回管理页 HTML。"""
        html = _INDEX_HTML
        return web.Response(text=html, content_type="text/html")

    app.router.add_get("/", index_handler)
    app.router.add_get("/api/list", list_handler)
    app.router.add_post("/api/upload", upload_handler)
    app.router.add_post("/api/description", description_handler)
    app.router.add_post("/api/rename", rename_handler)
    app.router.add_post("/api/delete", delete_handler)
    return app


_INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ComfyUI 工作流管理</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; margin: 0; padding: 16px; background: #1e1e1e; color: #d4d4d4; }
    h1 { font-size: 1.25rem; margin-bottom: 12px; }
    .upload { margin-bottom: 16px; padding: 12px; background: #252526; border-radius: 8px; }
    .upload input[type=file] { margin-right: 8px; }
    .upload button { padding: 6px 12px; cursor: pointer; background: #0e639c; color: #fff; border: none; border-radius: 4px; }
    .upload button:hover { background: #1177bb; }
    table { width: 100%; border-collapse: collapse; background: #252526; border-radius: 8px; overflow: hidden; }
    th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #333; }
    th { background: #2d2d30; }
    .desc { width: 40%; }
    .desc textarea { width: 100%; min-height: 48px; padding: 6px; background: #3c3c3c; color: #d4d4d4; border: 1px solid #555; border-radius: 4px; resize: vertical; }
    .act { white-space: nowrap; }
    .act button { margin-right: 6px; padding: 4px 10px; cursor: pointer; border: none; border-radius: 4px; font-size: 12px; }
    .btn-save { background: #0e639c; color: #fff; }
    .btn-rename { background: #5a5a5a; color: #fff; }
    .btn-del { background: #a1260d; color: #fff; }
    .msg { margin-top: 8px; padding: 8px; border-radius: 4px; }
    .msg.ok { background: #1e3a1e; color: #8bc34a; }
    .msg.err { background: #3a1e1e; color: #f44336; }
  </style>
</head>
<body>
  <h1>ComfyUI 工作流管理</h1>
  <div class="upload">
    <input type="file" id="fileInput" accept=".json">
    <button type="button" id="uploadBtn">上传 .json</button>
  </div>
  <div id="msg"></div>
  <table>
    <thead><tr><th>文件名</th><th class="desc">说明（供 LLM 选择工作流）</th><th class="act">操作</th></tr></thead>
    <tbody id="list"></tbody>
  </table>
  <script>
    const msg = (text, ok) => {
      const el = document.getElementById('msg');
      el.textContent = text;
      el.className = 'msg ' + (ok ? 'ok' : 'err');
    };
    const api = async (path, body) => {
      const res = await fetch(path, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {});
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || res.statusText);
      return data;
    };
    const loadList = async () => {
      const { files } = await api('/api/list');
      const tbody = document.getElementById('list');
      tbody.innerHTML = files.map(f => `
        <tr data-filename="${f.filename.replace(/"/g, '&quot;')}">
          <td>${f.filename}</td>
          <td class="desc"><textarea rows="2" placeholder="简要说明该工作流用途，供 LLM 选择">${(f.description || '').replace(/</g, '&lt;')}</textarea></td>
          <td class="act">
            <button class="btn-save">保存说明</button>
            <button class="btn-rename">重命名</button>
            <button class="btn-del">删除</button>
          </td>
        </tr>
      `).join('');
      tbody.querySelectorAll('.btn-save').forEach(btn => {
        btn.onclick = async () => {
          const row = btn.closest('tr');
          const filename = row.dataset.filename;
          const description = row.querySelector('textarea').value;
          try { await api('/api/description', { filename, description }); msg('已保存说明', true); } catch (e) { msg(e.message, false); }
        };
      });
      tbody.querySelectorAll('.btn-rename').forEach(btn => {
        btn.onclick = async () => {
          const row = btn.closest('tr');
          const old_name = row.dataset.filename;
          const new_name = prompt('新文件名（.json 结尾）', old_name);
          if (!new_name || new_name === old_name) return;
          try { await api('/api/rename', { old_name, new_name }); msg('已重命名', true); loadList(); } catch (e) { msg(e.message, false); }
        };
      });
      tbody.querySelectorAll('.btn-del').forEach(btn => {
        btn.onclick = async () => {
          if (!confirm('确定删除该工作流文件？')) return;
          const row = btn.closest('tr');
          const filename = row.dataset.filename;
          try { await api('/api/delete', { filename }); msg('已删除', true); loadList(); } catch (e) { msg(e.message, false); }
        };
      });
    };
    document.getElementById('uploadBtn').onclick = async () => {
      const input = document.getElementById('fileInput');
      if (!input.files.length) { msg('请选择文件', false); return; }
      const form = new FormData();
      form.append('file', input.files[0]);
      try {
        await fetch('/api/upload', { method: 'POST', body: form });
        msg('上传成功', true);
        input.value = '';
        loadList();
      } catch (e) { msg(e.message, false); }
    };
    loadList();
  </script>
</body>
</html>
"""


class ManagementServer:
    """
    工作流管理页服务器，支持 async start/stop，与 AstrBot 事件循环协同。
    参考 astrbot_plugin_stealer 的 WebServer 模式。
    """

    def __init__(
        self,
        workflows_dir: Path,
        meta_path: Path,
        load_meta: Callable[[], Dict[str, str]],
        save_meta: Callable[[Dict[str, str]], None],
    ):
        self.app = create_app(workflows_dir, meta_path, load_meta, save_meta)
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._started = False

    async def start(self, host: str, port: int) -> bool:
        """启动 Web 服务器。返回是否启动成功。"""
        try:
            self._runner = web.AppRunner(self.app, access_log=None)
            await self._runner.setup()
            self._site = web.TCPSite(self._runner, str(host).strip(), int(port))
            await self._site.start()
            self._started = True
            return True
        except OSError as e:
            if "Address already in use" in str(e) or getattr(e, "errno", None) in (98, 10048):
                raise RuntimeError(f"端口 {port} 已被占用，请更换端口或关闭占用程序") from e
            raise
        except Exception:
            raise

    async def stop(self) -> None:
        """停止 Web 服务器。"""
        if not self._started:
            return
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()
        self._site = None
        self._runner = None
        self._started = False
