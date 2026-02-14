# -*- coding: utf-8 -*-
"""
ComfyUI 工作流解析与执行引擎。
复用 nonebot_plugin_novelai 的工作流识别模式，使用 httpx 异步请求。
"""
import asyncio
import json
import random
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from astrbot.api import logger


def parse_workflow_filename(filename: str) -> Optional[Dict[str, Any]]:
    """
    解析工作流文件名，提取工作流名称和所需参数。
    格式示例：手办化+图片1.json, abc+文本2+图片2.json, 反推+图片1=文本1.json
    """
    if not filename.endswith(".json"):
        return None
    base_name = filename[:-5]
    output_texts = 0
    if "=" in base_name:
        parts_with_output = base_name.split("=")
        if len(parts_with_output) == 2:
            output_part = parts_with_output[1]
            if output_part.startswith("文本"):
                num_str = output_part[2:]
                if num_str.isdigit():
                    output_texts = int(num_str)
                    base_name = parts_with_output[0]
                else:
                    return None
            else:
                return None
        else:
            return None
    parts = base_name.split("+")
    if not parts:
        return None
    name = parts[0]
    texts = images = videos = 0
    seen_text = seen_image = seen_video = False
    for part in parts[1:]:
        if part.startswith("文本"):
            if seen_text:
                return None
            num_str = part[2:]
            if not num_str.isdigit():
                return None
            texts = int(num_str)
            seen_text = True
        elif part.startswith("图片"):
            if seen_image:
                return None
            num_str = part[2:]
            if not num_str.isdigit():
                return None
            images = int(num_str)
            seen_image = True
        elif part.startswith("视频"):
            if seen_video:
                return None
            num_str = part[2:]
            if not num_str.isdigit():
                return None
            videos = int(num_str)
            seen_video = True
        else:
            return None
    result = {
        "name": name,
        "texts": texts,
        "images": images,
        "videos": videos,
        "filename": filename,
    }
    if output_texts > 0:
        result["output_texts"] = output_texts
    return result


def list_workflows_in_dir(workflow_dir: Path) -> List[Dict[str, Any]]:
    """扫描指定目录下的 .json 工作流，返回可解析的工作流信息列表。"""
    workflows = []
    if not workflow_dir.exists():
        return workflows
    for f in workflow_dir.glob("*.json"):
        info = parse_workflow_filename(f.name)
        if info:
            workflows.append(info)
    return workflows


def find_workflow_file(
    workflow_name: str,
    text_count: int,
    image_count: int,
    video_count: int,
    workflow_dir: Path,
) -> Optional[str]:
    """根据工作流名称和参数数量在指定目录中查找最佳匹配的工作流文件路径。"""
    if not workflow_dir.exists():
        return None
    candidates = []
    for f in workflow_dir.glob("*.json"):
        info = parse_workflow_filename(f.name)
        if not info or info["name"] != workflow_name:
            continue
        rt, ri, rv = info["texts"], info["images"], info.get("videos", 0)
        if text_count >= rt and image_count >= ri and video_count >= rv:
            score = abs(text_count - rt) + abs(image_count - ri) + abs(video_count - rv)
            candidates.append({"file": str(f), "score": score})
    if not candidates:
        return None
    candidates.sort(key=lambda x: x["score"])
    return candidates[0]["file"]


class ComfyUIWorkflow:
    """异步执行 ComfyUI 工作流（使用 httpx）。"""

    def __init__(self, server_ip: str, client_id: str):
        self.server_ip = server_ip.rstrip("/").replace("http://", "").replace("https://", "")
        self.client_id = client_id
        self._base = f"http://{self.server_ip}"
        self._queue: deque = deque()
        self._processing = False

    def load_workflow_api(self, filepath: str) -> None:
        with open(filepath, "r", encoding="utf-8") as f:
            self.workflow_api = json.load(f)

    async def enqueue_workflow(
        self,
        base64_images: Optional[List[str]] = None,
        texts: Optional[List[str]] = None,
        videos: Optional[List[str]] = None,
        extract_text: bool = False,
    ) -> Tuple[Optional[str], str, List[str]]:
        """将任务加入队列并等待执行完成，返回 (文件URL, 文件类型, 文本输出列表)。"""
        base64_images = base64_images or []
        texts = texts or []
        videos = videos or []
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._queue.append((base64_images, texts, videos, extract_text, future))
        if not self._processing:
            asyncio.create_task(self._process_queue())
        return await future

    async def _process_queue(self) -> None:
        """单消费者：整段循环期间持锁，避免并发执行导致任务丢失/串线。"""
        self._processing = True
        try:
            while self._queue:
                base64_images, texts, videos, extract_text, future = self._queue.popleft()
                try:
                    result = await self._run_workflow(base64_images, texts, videos, extract_text)
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
        finally:
            self._processing = False

    def _replace_base64_images(self, data: Any, base64_images: List[str]) -> Tuple[Any, int]:
        """仅替换 class_type 为 ETN_LoadImageBase64 的节点（界面标题 Load Image (Base64)），传入 base64 到其 image 输入。其他一律不修改。"""
        counter = {"count": 0}

        def replace(d: Any) -> Any:
            if isinstance(d, dict):
                new_data = dict(d)
                inputs_modified = False
                if (
                    new_data.get("class_type") == "ETN_LoadImageBase64"
                    and counter["count"] < len(base64_images)
                ):
                    if "inputs" in new_data and isinstance(new_data["inputs"], dict) and "image" in new_data["inputs"]:
                        new_data["inputs"] = dict(new_data["inputs"])
                        new_data["inputs"]["image"] = base64_images[counter["count"]]
                        counter["count"] += 1
                        inputs_modified = True
                for k, v in d.items():
                    if k == "inputs" and inputs_modified:
                        continue
                    new_data[k] = replace(v)
                return new_data
            if isinstance(d, list):
                return [replace(x) for x in d]
            return d

        return replace(data), counter["count"]

    def _replace_video_nodes(self, data: Any, video_filenames: List[str]) -> Tuple[Any, int]:
        if not video_filenames:
            return data, 0
        counter = {"count": 0}
        broadcast = len(video_filenames) == 1

        def replace(d: Any) -> Any:
            if isinstance(d, dict):
                new_data = dict(d)
                inputs_modified = False
                if new_data.get("class_type") == "VHS_LoadVideo":
                    if "inputs" in new_data and isinstance(new_data["inputs"], dict) and "video" in new_data["inputs"]:
                        new_data["inputs"] = dict(new_data["inputs"])
                        if broadcast:
                            new_data["inputs"]["video"] = video_filenames[0]
                            counter["count"] += 1
                            inputs_modified = True
                        elif counter["count"] < len(video_filenames):
                            new_data["inputs"]["video"] = video_filenames[counter["count"]]
                            counter["count"] += 1
                            inputs_modified = True
                for k, v in d.items():
                    if k == "inputs" and inputs_modified:
                        continue
                    new_data[k] = replace(v)
                return new_data
            if isinstance(d, list):
                return [replace(x) for x in d]
            return d

        return replace(data), counter["count"]

    def _count_text_nodes(self, data: Any) -> int:
        """仅统计 Simple String 节点（inputs 含 text 或 string）。其他类型一律不计入。"""
        count = 0

        def walk(d: Any) -> None:
            nonlocal count
            if isinstance(d, dict):
                if d.get("class_type") == "Simple String" and isinstance(d.get("inputs"), dict):
                    if "text" in d["inputs"] or "string" in d["inputs"]:
                        count += 1
                for v in d.values():
                    walk(v)
            elif isinstance(d, list):
                for x in d:
                    walk(x)

        walk(data)
        return count

    def _smart_merge_texts(self, texts: List[str], slots: int) -> List[str]:
        if not texts or slots <= 0:
            return []
        if slots >= len(texts):
            return texts
        if slots == 1:
            return [" ".join(texts)]
        result = texts[: slots - 1]
        result.append(" ".join(texts[slots - 1 :]))
        return result

    def _update_text_nodes(self, data: Any, texts: List[str]) -> Tuple[Any, int]:
        """仅按顺序替换 Simple String 节点（text 或 string 输入）。其他类型一律不修改。"""
        slots = self._count_text_nodes(data)
        merged = self._smart_merge_texts(texts, slots)
        counter = {"count": 0}

        def replace(d: Any) -> Any:
            if isinstance(d, dict):
                new_data = dict(d)
                inputs_modified = False
                if (
                    new_data.get("class_type") == "Simple String"
                    and counter["count"] < len(merged)
                ):
                    if "inputs" in new_data and isinstance(new_data["inputs"], dict):
                        new_data["inputs"] = dict(new_data["inputs"])
                        if "text" in new_data["inputs"]:
                            new_data["inputs"]["text"] = merged[counter["count"]]
                            counter["count"] += 1
                            inputs_modified = True
                        elif "string" in new_data["inputs"]:
                            new_data["inputs"]["string"] = merged[counter["count"]]
                            counter["count"] += 1
                            inputs_modified = True
                for k, v in d.items():
                    # 若已在本层修改过 inputs，不要用 replace(v) 覆盖，否则会还原为空
                    if k == "inputs" and inputs_modified:
                        continue
                    new_data[k] = replace(v)
                return new_data
            if isinstance(d, list):
                return [replace(x) for x in d]
            return d

        result = replace(data)
        if counter["count"] > 0 and texts:
            logger.info(
                "[ComfyUI] Replaced %d Simple String node(s) with prompt: %s",
                counter["count"],
                texts[0][:80] + ("..." if len(texts[0]) > 80 else ""),
            )
        return result, counter["count"]

    def _randomize_seeds(self, data: Any) -> Any:
        if isinstance(data, dict):
            return {
                k: random.randint(1, 1000000000) if k in ("seed", "noise_seed") else self._randomize_seeds(v)
                for k, v in data.items()
            }
        if isinstance(data, list):
            return [self._randomize_seeds(x) for x in data]
        return data

    def _extract_text_outputs(
        self, workflow_data: Dict, history_data: Dict, prompt_id: str
    ) -> List[str]:
        text_outputs = []
        showtext_nodes = {
            nid: nd
            for nid, nd in workflow_data.items()
            if isinstance(nd, dict) and nd.get("class_type") == "ShowText|pysssss"
        }
        if not showtext_nodes:
            return text_outputs
        history_entry = history_data.get(prompt_id) if isinstance(history_data, dict) else None
        for node_id in showtext_nodes:
            text_content = None
            if history_entry and isinstance(history_entry, dict) and "prompts" in history_entry:
                prompts = history_entry["prompts"]
                if isinstance(prompts, list):
                    for item in prompts:
                        if isinstance(item, (list, tuple)) and len(item) >= 2 and str(item[0]) == node_id:
                            inp = item[1].get("inputs", {}) if isinstance(item[1], dict) else {}
                            text_content = inp.get("text_0")
                            break
            if text_content is None:
                nd = showtext_nodes[node_id]
                text_content = (nd.get("inputs") or {}).get("text_0")
            if isinstance(text_content, str):
                if "</think>" in text_content:
                    text_content = text_content.split("</think>", 1)[1]
                t = text_content.strip()
                if t:
                    text_outputs.append(t)
        return text_outputs

    async def _run_workflow(
        self,
        base64_images: List[str],
        texts: List[str],
        videos: List[str],
        extract_text: bool,
    ) -> Tuple[Optional[str], str, List[str]]:
        workflow_api_modified = json.loads(json.dumps(self.workflow_api))
        if base64_images:
            workflow_api_modified, _ = self._replace_base64_images(workflow_api_modified, base64_images)
        if videos:
            workflow_api_modified, _ = self._replace_video_nodes(workflow_api_modified, videos)
        if texts:
            workflow_api_modified, _ = self._update_text_nodes(workflow_api_modified, texts)
        workflow_api_modified = self._randomize_seeds(workflow_api_modified)

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{self._base}/prompt",
                json={"client_id": self.client_id, "prompt": workflow_api_modified},
            )
            resp.raise_for_status()
            prompt_id = resp.json()["prompt_id"]

        return await self._wait_and_collect_result(prompt_id, workflow_api_modified, extract_text)

    async def submit_only(
        self,
        base64_images: List[str],
        texts: List[str],
        videos: List[str],
        debug: bool = False,
    ) -> str:
        """
        仅提交工作流到 ComfyUI 队列，不等待完成。返回 prompt_id。
        用于由外部（如 query_wait）控制等待策略。
        debug=True 时在终端打印完整发送给 ComfyUI 的工作流 JSON 及文本替换信息。
        """
        workflow_api_modified = json.loads(json.dumps(self.workflow_api))
        text_slots = self._count_text_nodes(workflow_api_modified)
        if base64_images:
            workflow_api_modified, img_count = self._replace_base64_images(workflow_api_modified, base64_images)
            if debug:
                logger.info("[ComfyUI Debug] Replaced %d ETN_LoadImageBase64 node(s) with %d image(s)", img_count, len(base64_images))
        if videos:
            workflow_api_modified, _ = self._replace_video_nodes(workflow_api_modified, videos)
        if texts:
            workflow_api_modified, replaced = self._update_text_nodes(workflow_api_modified, texts)
            if debug:
                logger.info(
                    "[ComfyUI Debug] Simple String slots in workflow: %d, replaced: %d, texts passed: %s",
                    text_slots,
                    replaced,
                    texts,
                )
        workflow_api_modified = self._randomize_seeds(workflow_api_modified)
        if debug:
            try:
                payload = json.dumps(workflow_api_modified, ensure_ascii=False, indent=2)
                logger.info("[ComfyUI Debug] Full workflow JSON sent to ComfyUI (first 50k chars):\n%s", payload[:50000])
                if len(payload) > 50000:
                    logger.info("[ComfyUI Debug] ... (truncated, total %d chars)", len(payload))
            except Exception as e:
                logger.warning("[ComfyUI Debug] Failed to dump workflow: %s", e)
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{self._base}/prompt",
                json={"client_id": self.client_id, "prompt": workflow_api_modified},
            )
            if resp.status_code >= 400:
                try:
                    body = resp.text
                    if body:
                        logger.warning("[ComfyUI] /prompt %s response body: %s", resp.status_code, body[:2000])
                except Exception:
                    pass
            resp.raise_for_status()
            return resp.json()["prompt_id"]

    async def _wait_and_collect_result(
        self, prompt_id: str, workflow_api_modified: Dict, extract_text: bool
    ) -> Tuple[Optional[str], str, List[str]]:
        while True:
            async with httpx.AsyncClient(timeout=10.0) as client:
                queue_resp = await client.get(f"{self._base}/queue")
                queue_data = queue_resp.json()
            running = queue_data.get("queue_running", [])
            pending = queue_data.get("queue_pending", [])
            if not any(item[1] == prompt_id for item in running + pending):
                break
            await asyncio.sleep(1)
        async with httpx.AsyncClient(timeout=10.0) as client:
            history_resp = await client.get(f"{self._base}/history/{prompt_id}")
            image_info = history_resp.json()
        text_outputs = []
        if extract_text:
            text_outputs = self._extract_text_outputs(workflow_api_modified, image_info, prompt_id)
            text_outputs = [t.split("</think>", 1)[-1].strip() for t in text_outputs if t.strip()]
        if prompt_id not in image_info or "outputs" not in image_info[prompt_id]:
            return None, "unknown", text_outputs
        outputs = image_info[prompt_id]["outputs"]
        for key in outputs:
            out = outputs[key]
            if isinstance(out, dict) and "audio" in out:
                for audio in out["audio"]:
                    if audio.get("type") == "output":
                        fn = audio["filename"]
                        sub = audio.get("subfolder", "")
                        url = f"{self._base}/view?filename={fn}&subfolder={sub}&type=output" if sub else f"{self._base}/view?filename={fn}&type=output"
                        return url, "audio", text_outputs
            if isinstance(out, dict) and "gifs" in out:
                for video in out["gifs"]:
                    if video.get("type") == "output":
                        fn = video["filename"]
                        sub = video.get("subfolder", "")
                        url = f"{self._base}/view?filename={fn}&subfolder={sub}&type=output" if sub else f"{self._base}/view?filename={fn}&type=output"
                        return url, "video", text_outputs
            if isinstance(out, dict) and "images" in out:
                for img in out["images"]:
                    if img.get("type") == "output":
                        fn = img["filename"]
                        sub = img.get("subfolder", "")
                        url = f"{self._base}/view?filename={fn}&subfolder={sub}&type=output" if sub else f"{self._base}/view?filename={fn}&type=output"
                        return url, "image", text_outputs
        return None, "unknown", text_outputs
