# -*- coding: utf-8 -*-
"""
AstrBot ComfyUI 工作流插件（LLM 工具）。

- 任务标识：ComfyUI 提交后返回的 prompt_id（UUID）作为唯一任务 ID。
- 工具 comfyui_execute 返回 Task ID，comfyui_query_wait 可传 task_id 参数按该 ID 查询指定任务（跨轮次/跨会话）。
- 主逻辑见 main.py。
"""
