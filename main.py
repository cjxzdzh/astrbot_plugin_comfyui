# -*- coding: utf-8 -*-
"""
AstrBot ComfyUI 插件：将工作流封装为 LLM 工具，支持配置上传/管理、等待策略。
"""
import asyncio
import base64
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
import httpx
from astrbot.api import logger
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from pydantic import Field
from pydantic.dataclasses import dataclass

from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext

from .workflow_engine import (
    ComfyUIWorkflow,
    find_workflow_file,
    list_workflows_in_dir,
    parse_workflow_filename,
)

try:
    from astrbot.api import AstrBotConfig
except ImportError:
    AstrBotConfig = dict

# 插件数据目录（工作流文件与元数据）
PLUGIN_DATA_DIR = Path("data/plugin_data/astrbot_plugin_comfyui").resolve()
WORKFLOWS_DIR = PLUGIN_DATA_DIR / "workflows"
META_PATH = PLUGIN_DATA_DIR / "workflow_meta.json"

# 每个任务预估耗时（秒），用于等待策略
ESTIMATE_SECONDS_PER_JOB = 45
WAIT_THRESHOLD_SECONDS = 30

# 会话最近提交的任务：umo -> { "prompt_id", "server_ip", "client_id" }
# 同时写入 "default" 以便在工具内拿不到 event 时仍能查到当前会话任务
_session_pending: Dict[str, Dict[str, Any]] = {}

# 当前插件配置（由插件 __init__ 设置，供 LLM 工具读取）
_plugin_config: Any = None
# 插件 Context，供工具内调用 send_message 发送图片等
_plugin_context: Any = None


def _ensure_workflows_dir() -> None:
    WORKFLOWS_DIR.mkdir(parents=True, exist_ok=True)


def _load_workflow_meta() -> Dict[str, str]:
    """从 workflow_meta.json 读取 filename -> description。"""
    if not META_PATH.exists():
        return {}
    try:
        data = json.loads(META_PATH.read_text(encoding="utf-8"))
        return data.get("descriptions", data) if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_workflow_text_slots() -> Dict[str, List[str]]:
    """
    从 workflow_meta.json 读取 filename -> 文本槽位说明列表（与工作流中 Simple String 节点顺序一致）。
    用于 list_workflows 时告知 LLM 每个 text 的用途，例如 ["正面提示词", "负面提示词"] 或 ["修改说明"]。
    """
    if not META_PATH.exists():
        return {}
    try:
        data = json.loads(META_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        raw = data.get("text_slots")
        if not isinstance(raw, dict):
            return {}
        return {k: v if isinstance(v, list) else [] for k, v in raw.items()}
    except Exception:
        return {}


async def _load_workflow_descriptions(config: Any) -> Dict[str, str]:
    """工作流说明：优先从 workflow_meta.json 读取（管理页编辑），兼容旧配置 workflow_descriptions。"""
    meta = _load_workflow_meta()
    if meta:
        return meta
    raw = getattr(config, "workflow_descriptions", None) or "{}"
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw) if raw.strip() else {}
    except Exception:
        return {}


def _save_workflow_meta(descriptions: Dict[str, str]) -> None:
    """将 filename -> description 写入 workflow_meta.json。"""
    PLUGIN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(
        json.dumps({"descriptions": descriptions}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _get_workflow_dir() -> Path:
    """工作流目录：优先使用插件数据目录，若为空则回退到 sd_json（兼容旧路径）。"""
    _ensure_workflows_dir()
    if any(WORKFLOWS_DIR.glob("*.json")):
        return WORKFLOWS_DIR
    fallback = Path("sd_json")
    return fallback if fallback.exists() else WORKFLOWS_DIR


def _get_server_config(config: Any) -> tuple:
    server_ip = (getattr(config, "server_ip", None) or "127.0.0.1:8188").strip()
    client_id = (getattr(config, "client_id", None) or "astrbot-comfyui-1").strip()
    return server_ip, client_id


def _get_wait_threshold(config: Any) -> int:
    """从配置读取 query_wait 等待阈值（秒），未配置或非法时返回默认 30，并限制在 5～300 之间。"""
    raw = getattr(config, "wait_threshold_seconds", None) if not isinstance(config, dict) else config.get("wait_threshold_seconds")
    if raw is None:
        return WAIT_THRESHOLD_SECONDS
    try:
        n = int(raw)
        return max(5, min(300, n))
    except (TypeError, ValueError):
        return WAIT_THRESHOLD_SECONDS


def _get_session_key(context: Any) -> str:
    """从工具调用的 context 中解析会话 key（unified_msg_origin），拿不到时返回 'default' 以便仍能命中最近一次提交。"""
    try:
        ctx = getattr(context, "context", None)
        event = getattr(ctx, "event", None) if ctx else None
        if event is None and ctx is not None:
            event = getattr(getattr(ctx, "context", None), "event", None)
        if event is not None:
            umo = getattr(event, "unified_msg_origin", None) or ""
            if umo:
                return umo
    except Exception:
        pass
    return "default"


def _get_session_id_from_context(context: Any) -> Optional[str]:
    """从工具调用的 context 中解析 session_id，用于 send_message。"""
    try:
        ctx = getattr(context, "context", None)
        event = getattr(ctx, "event", None) if ctx else None
        if event is None and ctx is not None:
            event = getattr(getattr(ctx, "context", None), "event", None)
        if event is not None and hasattr(event, "get_session_id"):
            return event.get_session_id()
    except Exception:
        pass
    return None


async def _download_image_to_temp(image_url: str) -> Optional[str]:
    """
    将 ComfyUI 图片 URL 下载到临时文件。
    QQ 等平台无法访问 127.0.0.1，必须先下载再以本地文件形式发送。
    返回临时文件路径，失败返回 None。调用方负责在发送后删除临时文件。
    """
    if not image_url or not image_url.strip():
        return None
    url = image_url.strip()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.content
    except Exception as e:
        logger.warning("ComfyUI download image failed: %s", e)
        return None
    if not data:
        return None
    suffix = ".png"
    if b"JFIF" in data[:32] or b"\xff\xd8" in data[:2]:
        suffix = ".jpg"
    elif b"GIF" in data[:6]:
        suffix = ".gif"
    try:
        tmp_dir = PLUGIN_DATA_DIR / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        path = tmp_dir / f"comfyui_{uuid.uuid4().hex}{suffix}"
        async with aiofiles.open(path, "wb") as f:
            await f.write(data)
        return str(path)
    except Exception as e:
        logger.warning("ComfyUI write temp image failed: %s", e)
        return None


async def _send_image_to_session(session_id: str, image_url: str, plain_text: Optional[str] = None) -> bool:
    """
    向指定会话发送图片（可选带一句文本）。
    先将 ComfyUI 图片 URL 下载到临时文件，再用 Image.fromFileSystem + chain 发送，
    参考 astrbot_plugin_bilibili 的混合回复方式；发送后删除临时文件。
    """
    if not session_id or not image_url:
        return False
    ctx = _plugin_context
    if not ctx:
        return False
    temp_path = None
    try:
        temp_path = await _download_image_to_temp(image_url)
        if not temp_path or not Path(temp_path).exists():
            return False
        from astrbot.api.message_components import Image, Plain
        try:
            from astrbot.api.event import MessageEventResult
        except ImportError:
            from astrbot.core.message.message_event_result import MessageEventResult
        chain: List[Any] = []
        if plain_text and plain_text.strip():
            chain.append(Plain(plain_text.strip()))
        try:
            chain.append(Image.fromFileSystem(temp_path))
        except AttributeError:
            chain.append(Image.from_file_system(temp_path))
        if len(chain) == 1:
            result = MessageEventResult().image_result(temp_path)
        else:
            try:
                result = MessageEventResult(chain=chain)
            except TypeError:
                result = MessageEventResult().chain_result(chain)
        await ctx.send_message(session_id, result)
        return True
    except Exception as e:
        logger.warning("ComfyUI send image to session failed: %s", e)
        return False
    finally:
        if temp_path and Path(temp_path).exists():
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass


async def _get_queue_status(server_ip: str) -> tuple:
    """返回 (running_count, pending_count)，失败返回 (-1, -1)。"""
    base = f"http://{server_ip.lstrip('/').replace('http://', '').replace('https://', '')}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{base}/queue")
            data = r.json()
            running = len(data.get("queue_running", []))
            pending = len(data.get("queue_pending", []))
            return running, pending
    except Exception as e:
        logger.warning("get ComfyUI queue failed: %s", e)
        return -1, -1


async def _estimate_remaining_seconds(server_ip: str, prompt_id: str) -> int:
    """
    估算当前任务（prompt_id）完成还需多少秒。
    若已不在队列中则返回 0；否则用 (running+pending) * ESTIMATE_SECONDS_PER_JOB 粗估。
    """
    base = f"http://{server_ip.lstrip('/').replace('http://', '').replace('https://', '')}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{base}/queue")
            data = r.json()
            running = data.get("queue_running", [])
            pending = data.get("queue_pending", [])
            for item in running + pending:
                if len(item) >= 2 and item[1] == prompt_id:
                    # 还在队列中：粗略估计剩余时间
                    return (len(running) + len(pending)) * ESTIMATE_SECONDS_PER_JOB
    except Exception:
        pass
    return 0


async def _get_result_for_prompt(server_ip: str, prompt_id: str) -> tuple:
    """任务已完成时，从 history 拉取结果。返回 (file_url, file_type, text_outputs)。"""
    base = f"http://{server_ip.lstrip('/').replace('http://', '').replace('https://', '')}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            hist = await client.get(f"{base}/history/{prompt_id}")
            info = hist.json()
    except Exception:
        return None, "unknown", []
    if prompt_id not in info or "outputs" not in info[prompt_id]:
        return None, "unknown", []
    outputs = info[prompt_id]["outputs"]
    texts: List[str] = []
    for key in outputs:
        out = outputs[key]
        if isinstance(out, dict) and "audio" in out:
            for audio in out["audio"]:
                if audio.get("type") == "output":
                    fn, sub = audio["filename"], audio.get("subfolder", "")
                    url = f"{base}/view?filename={fn}&subfolder={sub}&type=output" if sub else f"{base}/view?filename={fn}&type=output"
                    return url, "audio", texts
        if isinstance(out, dict) and "gifs" in out:
            for video in out["gifs"]:
                if video.get("type") == "output":
                    fn, sub = video["filename"], video.get("subfolder", "")
                    url = f"{base}/view?filename={fn}&subfolder={sub}&type=output" if sub else f"{base}/view?filename={fn}&type=output"
                    return url, "video", texts
        if isinstance(out, dict) and "images" in out:
            for img in out["images"]:
                if img.get("type") == "output":
                    fn, sub = img["filename"], img.get("subfolder", "")
                    url = f"{base}/view?filename={fn}&subfolder={sub}&type=output" if sub else f"{base}/view?filename={fn}&type=output"
                    return url, "image", texts
    return None, "unknown", texts


async def _wait_for_completion(
    server_ip: str, client_id: str, prompt_id: str, timeout: int = 600
) -> tuple:
    """
    轮询直到任务完成，返回 (file_url, file_type, text_outputs)。
    超时或失败返回 (None, "unknown", [])。
    """
    base = f"http://{server_ip.lstrip('/').replace('http://', '').replace('https://', '')}"
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"{base}/queue")
                data = r.json()
                running = data.get("queue_running", [])
                pending = data.get("queue_pending", [])
                if not any(item[1] == prompt_id for item in running + pending):
                    break
        except Exception:
            pass
        await asyncio.sleep(2)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            hist = await client.get(f"{base}/history/{prompt_id}")
            info = hist.json()
    except Exception:
        return None, "unknown", []
    if prompt_id not in info or "outputs" not in info[prompt_id]:
        return None, "unknown", []
    outputs = info[prompt_id]["outputs"]
    texts: List[str] = []
    for key in outputs:
        out = outputs[key]
        if isinstance(out, dict) and "audio" in out:
            for audio in out["audio"]:
                if audio.get("type") == "output":
                    fn, sub = audio["filename"], audio.get("subfolder", "")
                    url = f"{base}/view?filename={fn}&subfolder={sub}&type=output" if sub else f"{base}/view?filename={fn}&type=output"
                    return url, "audio", texts
        if isinstance(out, dict) and "gifs" in out:
            for video in out["gifs"]:
                if video.get("type") == "output":
                    fn, sub = video["filename"], video.get("subfolder", "")
                    url = f"{base}/view?filename={fn}&subfolder={sub}&type=output" if sub else f"{base}/view?filename={fn}&type=output"
                    return url, "video", texts
        if isinstance(out, dict) and "images" in out:
            for img in out["images"]:
                if img.get("type") == "output":
                    fn, sub = img["filename"], img.get("subfolder", "")
                    url = f"{base}/view?filename={fn}&subfolder={sub}&type=output" if sub else f"{base}/view?filename={fn}&type=output"
                    return url, "image", texts
    return None, "unknown", texts


async def _extract_images_from_event_async(event: Any) -> List[str]:
    """异步从事件中提取图片 base64。"""
    base64_list: List[str] = []
    try:
        msg_obj = getattr(event, "message_obj", None)
        if not msg_obj:
            return base64_list
        chain = getattr(msg_obj, "message", None) or []
        async with httpx.AsyncClient(timeout=30.0) as client:
            for comp in chain:
                comp_type = getattr(comp, "type", None) or (comp.get("type") if isinstance(comp, dict) else None)
                if comp_type in ("image", "Image"):
                    url = getattr(comp, "url", None) or (comp.get("url") if isinstance(comp, dict) else None)
                    if not url:
                        file_path = getattr(comp, "file", None) or (comp.get("file") if isinstance(comp, dict) else None)
                        if file_path and Path(file_path).exists():
                            async with aiofiles.open(file_path, "rb") as f:
                                data = await f.read()
                            base64_list.append(base64.b64encode(data).decode("utf-8"))
                        continue
                    try:
                        resp = await client.get(url.replace("\n", ""))
                        if resp.status_code == 200:
                            base64_list.append(base64.b64encode(resp.content).decode("utf-8"))
                    except Exception as e:
                        logger.warning("download image for tool failed: %s", e)
    except Exception as e:
        logger.warning("extract images from event failed: %s", e)
    return base64_list


def _is_allowed_local_image_path(file_path: Path) -> bool:
    """
    仅允许插件数据目录内的本地路径，防止路径穿越（如 ../../../etc/passwd）。
    不在允许范围内的路径一律拒绝，不跨地址、不穿越。
    """
    try:
        resolved = file_path.resolve()
        base = PLUGIN_DATA_DIR.resolve()
        # 必须位于 base 之下（含 base 自身）；禁止 .. 逃逸
        return resolved == base or str(resolved).startswith(str(base) + os.sep)
    except Exception:
        return False


def _parse_comfyui_400_summary(body: str) -> Optional[str]:
    """
    解析 ComfyUI /prompt 返回的 400 JSON，生成给 LLM 看的简短说明。
    例如：工作流里用的模型在服务器上不存在（value_not_in_list, ckpt_name）。
    """
    if not body or not body.strip():
        return None
    try:
        data = json.loads(body)
    except Exception:
        return None
    node_errors = data.get("node_errors") if isinstance(data, dict) else None
    if not isinstance(node_errors, dict):
        return None
    parts = []
    for _node_id, node_data in node_errors.items():
        if not isinstance(node_data, dict):
            continue
        err_list = node_data.get("errors")
        if not isinstance(err_list, list):
            continue
        for err in err_list:
            if not isinstance(err, dict):
                continue
            if err.get("type") == "value_not_in_list":
                details = err.get("details") or ""
                extra = err.get("extra_info") or {}
                input_name = extra.get("input_name", "")
                received = extra.get("received_value", "")
                config_list = extra.get("input_config")
                if isinstance(config_list, list) and len(config_list) and isinstance(config_list[0], list):
                    allowed = config_list[0][:10]
                else:
                    allowed = []
                if input_name == "ckpt_name" and received:
                    allowed_str = "、".join(allowed) if allowed else "(见服务器模型目录)"
                    parts.append(
                        f"工作流中使用的模型 '{received}' 在当前 ComfyUI 服务器上不存在；"
                        f"服务器可用模型包括：{allowed_str}。请改用「改图」等其它工作流，或在该工作流中把模型改为已有模型。"
                    )
                    break
                if not parts and details:
                    parts.append(f"ComfyUI 校验失败: {details[:500]}")
    return " ".join(parts) if parts else None


async def _image_sources_to_base64(sources: List[str]) -> List[str]:
    """
    将「图片来源」列表转为 base64 列表，支持两种来源（占位符方案，由插件拉取/读取，LLM 不生成 base64）：
    - 服务器 URL：如 QQ 多媒体链接，插件下载后转 base64；
    - 本地路径：仅允许插件数据目录内的路径，否则拒绝（防路径穿越）。
    用于 comfyui_execute 的 image_urls 参数（可传 URL 或允许范围内的本地路径）。
    """
    result: List[str] = []
    for s in sources:
        if not s or not isinstance(s, str):
            continue
        s = s.strip()
        # 本地文件路径：仅允许在 PLUGIN_DATA_DIR 内，禁止 ../ 穿越
        if not s.startswith("http"):
            p = Path(s)
            if not p.exists() or not p.is_file():
                continue
            if not _is_allowed_local_image_path(p):
                logger.warning("ComfyUI rejected local image path (outside allowed dir, path traversal not allowed): %s", s)
                continue
            try:
                async with aiofiles.open(p, "rb") as f:
                    data = await f.read()
                result.append(base64.b64encode(data).decode("utf-8"))
            except Exception as e:
                logger.warning("ComfyUI read local image failed: %s", e)
            continue
        # 服务器 URL
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(s.replace("\n", ""))
                if resp.status_code == 200 and resp.content:
                    result.append(base64.b64encode(resp.content).decode("utf-8"))
        except Exception as e:
            logger.warning("ComfyUI fetch image_url failed: %s", e)
    return result


# --------------- LLM Tools ---------------


@dataclass
class ComfyUIListWorkflowsTool(FunctionTool[AstrAgentContext]):
    """查询当前可用的 ComfyUI 工作流列表及说明、所需参数，供 LLM 选择工作流时使用。"""

    name: str = "comfyui_list_workflows"
    description: str = (
        "Query the list of available ComfyUI workflows. Returns workflow name, user-defined description, "
        "and required inputs (texts/images/videos). Use this before comfyui_execute. When calling execute, generate texts according to each workflow's description and text_slots (e.g. '根据图2的XX修改图1'), not just image content description."
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {},
            "required": [],
        }
    )

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> ToolExecResult:
        logger.info("[ComfyUI Tool] comfyui_list_workflows called with args: %s", kwargs)
        config = _plugin_config
        if not config:
            return "Plugin config not available."
        server_ip, _ = _get_server_config(config)
        descriptions = await _load_workflow_descriptions(config)
        wf_dir = _get_workflow_dir()
        workflows = list_workflows_in_dir(wf_dir)
        if not workflows:
            return "No workflow files found. Please upload workflow JSON files to the plugin workflows directory or use /comfyui upload."
        lines = [
            "Available workflows. When calling comfyui_execute you MUST pass the required inputs:",
            "- workflow_name: exact name below; texts: array of strings; images: from user message auto-used, OR pass image_urls (URL or local path) when message has no image; videos: array of .mp4 filenames if needed.",
            "- Important: texts must be generated according to each workflow's description and text_slots below (e.g. '根据图2的XX修改图1'), NOT just the raw image content description from vision—read the workflow description and produce the required style of text.",
            "",
        ]
        text_slots_map = _load_workflow_text_slots()
        for w in workflows:
            name = w["name"]
            desc = descriptions.get(w["filename"], "") or "(no description)"
            t, i, v = w["texts"], w["images"], w.get("videos", 0)
            req_parts = []
            if t > 0:
                slots = text_slots_map.get(w["filename"])
                if isinstance(slots, list) and len(slots) >= t:
                    slot_desc = ", ".join(slots[:t])
                    req_parts.append(f"{t} text(s) (required, in order: {slot_desc})")
                else:
                    req_parts.append(f"{t} text(s) (required, pass as texts array)")
            if i > 0:
                req_parts.append(f"{i} image(s) (required: from user message, or pass image_urls: URL or local path, e.g. from get_image_from_context)")
            if v > 0:
                req_parts.append(f"{v} video(s) (required, pass as videos array)")
            req_str = "; ".join(req_parts) if req_parts else "no extra inputs"
            lines.append(f"- {name}: {desc}. Required: {req_str}")
        return "\n".join(lines)


@dataclass
class ComfyUIStatusTool(FunctionTool[AstrAgentContext]):
    """
    查询 ComfyUI 队列状态。
    若本会话有已提交任务：预计剩余 < 配置阈值则阻塞至完成后返回；>= 阈值则先等待阈值秒再返回剩余时间。阈值由插件配置 wait_threshold_seconds 决定。
    """

    name: str = "comfyui_status"
    description: str = (
        "Get ComfyUI queue status (running/pending counts). "
        "If this session has a submitted task: if estimated wait below threshold blocks until done; "
        "otherwise waits threshold seconds then returns remaining time. Threshold is configurable in plugin settings."
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {},
            "required": [],
        }
    )

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> ToolExecResult:
        logger.info("[ComfyUI Tool] comfyui_status called with args: %s", kwargs)
        config = _plugin_config
        if not config:
            return "Plugin config not available."
        wait_threshold = _get_wait_threshold(config)
        server_ip, _ = _get_server_config(config)
        session_key = _get_session_key(context.context)
        pending = _session_pending.get(session_key) or _session_pending.get("default")
        if pending and pending.get("prompt_id") and server_ip:
            remaining = await _estimate_remaining_seconds(server_ip, pending["prompt_id"])
            if remaining == 0:
                url, ftype, texts = await _get_result_for_prompt(server_ip, pending["prompt_id"])
                for k in list(_session_pending.keys()):
                    if _session_pending.get(k) == pending:
                        _session_pending.pop(k, None)
                if url:
                    if ftype == "image":
                        session_id = _get_session_id_from_context(context.context)
                        if session_id:
                            await _send_image_to_session(session_id, url, "图好了～")
                        return f"Task completed. Output: image. Image has been sent to the user. IMAGE_URL: {url} Queue: 0 running, 0 pending."
                    return f"Task completed. Output: {ftype}. URL: {url} Queue: 0 running, 0 pending."
                return "Task completed (no output file). Queue: 0 running, 0 pending."
            if remaining < wait_threshold:
                client_id = pending.get("client_id", "")
                url, ftype, texts = await _wait_for_completion(
                    server_ip, client_id, pending["prompt_id"], timeout=remaining + 120
                )
                for k in list(_session_pending.keys()):
                    if _session_pending.get(k) == pending:
                        _session_pending.pop(k, None)
                if url:
                    if ftype == "image":
                        session_id = _get_session_id_from_context(context.context)
                        if session_id:
                            await _send_image_to_session(session_id, url, "图好了～")
                        return f"Task completed. Output: image. Image has been sent to the user. IMAGE_URL: {url} Queue: 0 running, 0 pending."
                    return f"Task completed. Output: {ftype}. URL: {url} Queue: 0 running, 0 pending."
                return "Task finished. Queue: 0 running, 0 pending."
            await asyncio.sleep(wait_threshold)
            running, pending_count = await _get_queue_status(server_ip)
            remaining_after = await _estimate_remaining_seconds(server_ip, pending["prompt_id"])
            return (
                f"ComfyUI queue: {running} running, {pending_count} pending. "
                f"Your task estimated remaining: about {remaining_after} seconds. Call again to re-check."
            )
        running, pending_count = await _get_queue_status(server_ip)
        if running < 0:
            return "ComfyUI server unreachable. Please check server_ip and network."
        return f"ComfyUI queue: {running} running, {pending_count} pending."


@dataclass
class ComfyUIQueryWaitTool(FunctionTool[AstrAgentContext]):
    """
    查询当前会话已提交任务的等待情况。
    若预计剩余 < 配置阈值则阻塞直到完成并返回结果；>= 阈值则先等待阈值秒再返回剩余时间。阈值由插件配置 wait_threshold_seconds 决定。
    """

    name: str = "comfyui_query_wait"
    description: str = (
        "Query the wait status of the last submitted ComfyUI task in this session. "
        "If estimated remaining wait below threshold, blocks until completion and returns result; "
        "otherwise waits threshold seconds then returns remaining time. Threshold configurable in plugin. Call after comfyui_execute."
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {},
            "required": [],
        }
    )

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> ToolExecResult:
        logger.info("[ComfyUI Tool] comfyui_query_wait called with args: %s", kwargs)
        wait_threshold = _get_wait_threshold(_plugin_config or {})
        session_key = _get_session_key(context.context)
        pending = _session_pending.get(session_key) or _session_pending.get("default")
        if not pending:
            return "No pending ComfyUI task in this session. Submit a workflow with comfyui_execute first."
        prompt_id = pending.get("prompt_id")
        server_ip = pending.get("server_ip")
        if not prompt_id or not server_ip:
            return "Invalid pending task data."
        remaining = await _estimate_remaining_seconds(server_ip, prompt_id)
        if remaining == 0:
            url, ftype, texts = await _get_result_for_prompt(server_ip, prompt_id)
            for k in list(_session_pending.keys()):
                if _session_pending.get(k) == pending:
                    _session_pending.pop(k, None)
            if url:
                if ftype == "image":
                    session_id = _get_session_id_from_context(context.context)
                    if session_id:
                        await _send_image_to_session(session_id, url, "图好了～")
                extra = (" Text output: " + "; ".join(texts)) if texts else ""
                if ftype == "image":
                    return f"Task completed. Output: image. Image has been sent to the user. IMAGE_URL: {url}{extra}"
                return f"Task completed. Output: {ftype}. URL: {url}{extra}"
            return "Task completed but no output file found."
        if remaining < wait_threshold:
            client_id = pending.get("client_id", "")
            url, ftype, texts = await _wait_for_completion(server_ip, client_id, prompt_id, timeout=remaining + 120)
            for k in list(_session_pending.keys()):
                if _session_pending.get(k) == pending:
                    _session_pending.pop(k, None)
            if url:
                if ftype == "image":
                    session_id = _get_session_id_from_context(context.context)
                    if session_id:
                        await _send_image_to_session(session_id, url, "图好了～")
                extra = (" Text output: " + "; ".join(texts)) if texts else ""
                if ftype == "image":
                    return f"Task completed. Output: image. Image has been sent to the user. IMAGE_URL: {url}{extra}"
                return f"Task completed. Output: {ftype}. URL: {url}{extra}"
            return "Task finished but no output file."
        await asyncio.sleep(wait_threshold)
        remaining_after = await _estimate_remaining_seconds(server_ip, prompt_id)
        return f"Still in queue. Estimated remaining wait: about {remaining_after} seconds. Call comfyui_query_wait again to re-check or get result when ready."


@dataclass
class ComfyUIExecuteTool(FunctionTool[AstrAgentContext]):
    """
    执行指定的 ComfyUI 工作流。工作流名称需与 list_workflows 返回的 name 一致。
    文本参数通过 texts 传入；图片从当前会话消息中自动提取；若工作流需要图而消息无图，可传 image_urls（占位符），插件会下载并转 base64 注入。
    """

    name: str = "comfyui_execute"
    description: str = (
        "Execute a ComfyUI workflow by name. Use comfyui_list_workflows first to get valid names and required inputs. "
        "Pass text arguments as texts array. Texts must follow the workflow's description and text_slots from comfyui_list_workflows (e.g. if workflow requires '根据图2的XX修改图1', generate that style of instruction from user intent, not just image content description). Images: auto-used from current message if user sent any; if workflow needs images but message has none, pass image_urls (URL or local path); plugin downloads/reads and injects as base64."
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "workflow_name": {
                    "type": "string",
                    "description": "Exact workflow name (e.g. from comfyui_list_workflows).",
                },
                "texts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Required when workflow needs text. Content must match workflow description and text_slots (see comfyui_list_workflows)—e.g. modification instruction like '根据图2的XX修改图1', not just image content description. Generate according to workflow requirement and user intent.",
                },
                "videos": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of video filenames (.mp4) on server for video workflows.",
                },
                "image_urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Placeholder for images: when workflow needs images (e.g. 改图) but current message has none, pass image source(s) here: server URL (e.g. from get_image_from_context) or local file path (e.g. QQ local cache). Plugin downloads/reads and injects as base64. Order: message images first, then image_urls.",
                },
            },
            "required": ["workflow_name"],
        }
    )

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> ToolExecResult:
        logger.info(
            "[ComfyUI Tool] comfyui_execute called with args: workflow_name=%r, texts=%r, videos=%r, image_urls=%r, full_kwargs=%s",
            kwargs.get("workflow_name"),
            kwargs.get("texts"),
            kwargs.get("videos"),
            kwargs.get("image_urls"),
            kwargs,
        )
        workflow_name = (kwargs.get("workflow_name") or "").strip()
        texts = kwargs.get("texts") or []
        videos = list(kwargs.get("videos") or [])
        image_urls_arg = kwargs.get("image_urls") or []
        if isinstance(image_urls_arg, str):
            image_urls_arg = [image_urls_arg]
        image_urls_arg = [u for u in image_urls_arg if u and isinstance(u, str)]
        if not workflow_name:
            return "workflow_name is required."
        config = _plugin_config
        if not config:
            return "Plugin config not available."
        server_ip, client_id = _get_server_config(config)
        wf_dir = _get_workflow_dir()
        ctx = getattr(context.context, "context", None)
        event = getattr(ctx, "event", None) if ctx else None
        images_b64 = await _extract_images_from_event_async(event) if event else []
        if image_urls_arg:
            from_sources = await _image_sources_to_base64(image_urls_arg)
            images_b64.extend(from_sources)
            if from_sources:
                logger.info("[ComfyUI Tool] Injected %d image(s) from image_urls placeholder (URL or local path).", len(from_sources))
        workflow_file = find_workflow_file(
            workflow_name, len(texts), len(images_b64), len(videos), wf_dir
        )
        if not workflow_file:
            hint = ""
            if len(images_b64) == 0:
                hint = (
                    " Current message has no image (images=0). If the user already sent an image, "
                    "call get_image_from_context first to get the image URL, then call comfyui_execute again with image_urls=[that URL]."
                )
            return (
                f"No matching workflow for name '{workflow_name}' with texts={len(texts)}, images={len(images_b64)}, videos={len(videos)}. "
                "Use comfyui_list_workflows to see available workflows and required inputs. "
                "Possible reasons: (1) workflow name typo—use exact name from list; (2) too few texts/images/videos—check required counts for that workflow."
                + hint
            )
        info = parse_workflow_filename(Path(workflow_file).name)
        if not info:
            return "Workflow file format error: workflow filename does not match expected pattern (name+文本N+图片M.json)."
        wf_filename = Path(workflow_file).name
        descriptions = await _load_workflow_descriptions(config)
        workflow_desc = (descriptions.get(wf_filename) or "").strip()
        desc_reminder = ""
        if workflow_desc:
            desc_reminder = (
                f"\n\n[工作流「{workflow_name}」说明 (下次调用请按此生成 texts): {workflow_desc}"
                "\n文本须按上述说明填写（如「根据图2的XX修改图1」），不要只传图片内容描述。]"
            )
        if len(images_b64) < info["images"] or len(texts) < info["texts"] or len(videos) < info.get("videos", 0):
            hint = ""
            if len(images_b64) < info["images"] and len(images_b64) == 0:
                hint = (
                    " No image in current message. If the user already sent an image, "
                    "call get_image_from_context to get the image URL, then call comfyui_execute again with image_urls=[that URL]."
                )
            return (
                f"Workflow '{workflow_name}' requires at least texts={info['texts']}, images={info['images']}, videos={info.get('videos', 0)}. "
                f"Provided: texts={len(texts)}, images={len(images_b64)}, videos={len(videos)}. "
                "Pass more texts (as texts array), or provide images (from message / image_urls), or videos as needed."
                + hint
                + desc_reminder
            )
        try:
            debug = bool(getattr(config, "debug_mode", False) if not isinstance(config, dict) else config.get("debug_mode", False))
            workflow = ComfyUIWorkflow(server_ip, client_id)
            workflow.load_workflow_api(workflow_file)
            prompt_id = await workflow.submit_only(images_b64, texts, videos, debug=debug)
            session_key = _get_session_key(context.context)
            pending_data = {
                "prompt_id": prompt_id,
                "server_ip": server_ip,
                "client_id": client_id,
            }
            _session_pending[session_key] = pending_data
            if session_key != "default":
                _session_pending["default"] = pending_data
            return (
                f"Workflow '{workflow_name}' submitted. Task ID: {prompt_id}. "
                "Call comfyui_query_wait to wait for completion or get remaining wait time (threshold configurable in plugin)."
                + desc_reminder
            )
        except httpx.HTTPStatusError as e:
            body = ""
            try:
                if e.response is not None:
                    body = e.response.text
            except Exception:
                pass
            summary = _parse_comfyui_400_summary(body)
            msg = (
                f"Execute failed: ComfyUI returned {e.response.status_code if e.response else '?'}. "
                + (summary if summary else (f"Server message: {body[:1500]}" if body else str(e)))
            )
            logger.exception("comfyui_execute failed: %s", msg)
            return msg + (" Suggest: fix workflow or use another workflow that runs on this server." if summary else " Possible causes: workflow node/input mismatch, invalid image format, or server error.") + desc_reminder
        except Exception as e:
            logger.exception("comfyui_execute failed")
            return (
                f"Execute failed: {e}. "
                "Possible causes: ComfyUI server unreachable or timeout; workflow node error; invalid input. "
                "Check server address and that the workflow JSON is valid."
                + desc_reminder
            )


# --------------- Plugin ---------------


@register(
    "comfyui",
    "ComfyUI",
    "ComfyUI 工作流 LLM 工具：执行/查询工作流/等待查询/状态；支持配置上传与工作流说明",
    "1.0.0",
    "",
)
class ComfyUIPlugin(Star):
    def __init__(self, context: Context, config: Any = None):
        super().__init__(context)
        global _plugin_config, _plugin_context
        _plugin_config = self.config = config or {}
        _plugin_context = self.context
        self.context.add_llm_tools(
            ComfyUIListWorkflowsTool(),
            ComfyUIStatusTool(),
            ComfyUIQueryWaitTool(),
            ComfyUIExecuteTool(),
        )
        self._web_server = None  # ManagementServer 实例，在 initialize 中启动

    async def initialize(self) -> None:
        """插件加载完成后启动工作流管理页（若启用）。"""
        config = self.config or {}
        enabled = bool(getattr(config, "webui_enabled", True))
        if not enabled:
            logger.info("ComfyUI 工作流管理页已禁用")
            return
        try:
            from .management_server import ManagementServer
        except ImportError as e:
            logger.warning("ComfyUI 管理页不可用（请安装 aiohttp）: %s", e)
            return
        host = (getattr(config, "webui_host", None) or "127.0.0.1").strip()
        port = int(getattr(config, "webui_port", 6187) or 6187)
        try:
            self._web_server = ManagementServer(
                workflows_dir=WORKFLOWS_DIR,
                meta_path=META_PATH,
                load_meta=_load_workflow_meta,
                save_meta=_save_workflow_meta,
            )
            await self._web_server.start(host, port)
            if host == "0.0.0.0":
                logger.info(
                    "ComfyUI 工作流管理页已启动，监听 0.0.0.0:%s（本机访问 http://127.0.0.1:%s）",
                    port,
                    port,
                )
            else:
                logger.info("ComfyUI 工作流管理页已启动: http://%s:%s", host, port)
        except Exception as e:
            logger.error("启动 ComfyUI 工作流管理页失败: %s", e, exc_info=True)
            self._web_server = None

    async def terminate(self) -> None:
        """插件卸载时关闭工作流管理页。"""
        if getattr(self, "_web_server", None):
            try:
                await self._web_server.stop()
                logger.info("ComfyUI 工作流管理页已关闭")
            except Exception as e:
                logger.warning("关闭 ComfyUI 工作流管理页时出错: %s", e)
            self._web_server = None

    @filter.command("comfyui")
    async def cmd_comfyui(self, event: AstrMessageEvent):
        """ComfyUI 插件：使用 /comfyui 查询 或 回复一条包含 JSON 文件的消息后发送 /comfyui 上传"""
        msg = (event.message_str or "").strip()
        if msg == "查询" or msg == "list" or msg == "help":
            wf_dir = _get_workflow_dir()
            workflows = list_workflows_in_dir(wf_dir)
            descriptions = await _load_workflow_descriptions(self.config)
            if not workflows:
                yield event.plain_result("当前没有工作流文件。请使用 /comfyui 上传 并回复一条包含 .json 文件的消息。")
                return
            lines = ["工作流列表："]
            for w in workflows:
                desc = descriptions.get(w["filename"], "") or "(未填写说明)"
                lines.append(f"- {w['name']} ({w['filename']}): {desc}")
            yield event.plain_result("\n".join(lines))
            return
        if msg == "上传" or msg == "upload":
            # 从当前消息或回复中取第一个 .json 文件
            chain = getattr(getattr(event, "message_obj", None), "message", None) or []
            reply = getattr(event, "reply", None)
            if reply:
                reply_chain = getattr(getattr(reply, "message_obj", None), "message", None) or getattr(reply, "message", None) or []
                chain = list(reply_chain) + list(chain)
            file_url = None
            file_name = None
            for comp in chain:
                ctype = getattr(comp, "type", None) or (comp.get("type") if isinstance(comp, dict) else None)
                if ctype in ("file", "File", "image", "Image"):
                    url = getattr(comp, "url", None) or (comp.get("url") if isinstance(comp, dict) else None)
                    name = getattr(comp, "name", None) or getattr(comp, "filename", None) or (comp.get("name") or comp.get("filename") if isinstance(comp, dict) else None)
                    if url and name and str(name).endswith(".json"):
                        file_url = url
                        file_name = name
                        break
            if not file_url:
                yield event.plain_result("请回复一条包含 .json 工作流文件的消息，然后发送 /comfyui 上传。")
                return
            _ensure_workflows_dir()
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    r = await client.get(file_url.replace("\n", ""))
                    r.raise_for_status()
                out_path = WORKFLOWS_DIR / (file_name or "workflow.json")
                async with aiofiles.open(out_path, "wb") as f:
                    await f.write(r.content)
                yield event.plain_result(
                    f"已保存工作流到 {out_path.name}。"
                    "请在「工作流管理页」（配置中启用 webui_enabled 并设置 webui_port 后访问对应地址）为该文件填写说明，供 LLM 选择。"
                )
            except Exception as e:
                logger.exception("comfyui upload failed")
                yield event.plain_result(f"上传失败: {e}")
            return
        yield event.plain_result("用法：/comfyui 查询 | /comfyui 上传（需回复含 .json 的消息）")
