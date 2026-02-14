# AstrBot ComfyUI 工作流插件

将 ComfyUI 工作流封装为 LLM 可调用的工具。**向 BOT 描述你的目的，它会自动检查可用工作流并调用，最终将图片/视频等产物发送给你。**

> 已验证 **Ubuntu Server** 上部署可用。

---


### 适用场景

- **文生图**：用文字描述，让 BOT 根据可用工作流生成图片
- **文生视频**：用文字描述，生成视频
- **图文改图**：发图 + 修改说明，让 BOT 按你的要求修改图片

### 使用方式

直接向 BOT 说出你的需求即可，无需记忆命令或工作流名。BOT 会：

1. 自动检查当前有哪些可用工作流
2. 根据你的描述选择合适的工作流并调用
3. 将生成的图片或视频发送给你

### 使用示例

| 你说什么 | 说明 |
|----------|------|
| 帮我画一张猫猫动漫图 | 文生图，BOT 会选用文生图工作流并生成 |
| （引用一条带图的消息）帮我将黑丝改成白丝 | 图文改图，BOT 会取被引用的图并按说明修改 |
| 之前画的白丝图，脱了吧 | 引用之前 BOT 发的图，继续修改；BOT 会找到该图并调用改图工作流 |

---

## 目标功能（技术说明）

**任意**在 ComfyUI 上能跑通的工作流，只要把「需要由 LLM/用户传入」的**文本、图片、视频**入口，换成约定好的几类节点（Simple String、ETN_LoadImageBase64、VHS_LoadVideo），即可接入 AstrBot，由 LLM 自动注入参数并执行。

- **约定**：可注入的入口仅限上述三类节点；工作流里其他逻辑（模型、采样、ControlNet、多步推理等）一律保持原样。
- **流程**：在 ComfyUI 中设计好工作流 → 导出 API 格式 JSON → 按规范命名（见第四节）→ 上传到本插件并填写说明，即可被 LLM 选用并调用。

---

## 一、依赖说明

### 1.1 AstrBot 插件依赖（可选但推荐）

- **[astrbot_plugin_qq_tools](https://github.com/YUMU1658/astrbot_plugin_qq_tools)**（工具名均为 `qts_` 开头）
  - 提供 **`qts_get_recent_messages`**、**`qts_get_message_detail`** 等：获取最近消息列表及单条消息详情。
  - 当用户引用上一条消息中的图/视频（如「把这张图改成雨天」）时，LLM 可先调用 **`qts_get_recent_messages`** 找到对应消息，再用 **`qts_get_message_detail`** 取详情；消息 content 中会包含本插件写入的 **「ComfyUI 图片路径: /path」** / **「ComfyUI 视频路径: /path」**，将该路径传入 `comfyui_execute` 的 **`image_urls`** 即可。**获取最近媒体信息时优先使用 qts_get_recent_messages。**

- **[astrbot_plugin_image_url_base64_to_mcp](https://github.com/Thetail001/astrbot_plugin_image_url_base64_to_mcp)**
  - 提供 LLM 工具 **`get_image_from_context`**：从对话上下文中获取用户发送的图片（URL 或 base64）。
  - 当用户说「改这张图」但当前消息里未带图、或平台未把图片注入到本插件可读的 message 时，LLM 可先调用 `get_image_from_context` 拿到图片 URL，再在 `comfyui_execute` 中传入 **`image_urls=[该 URL]`**，由本插件下载并转为 base64 注入工作流。
  - 不安装该插件也可使用本插件，但「引用之前消息里的图」进行改图时，需依赖该工具或上述 qts 工具才能稳定拿到图片。

### 1.2 Python 依赖

- 见 **`requirements.txt`**（如 httpx、aiofiles）。将本插件放入 AstrBot 的 `data/plugins/` 后，按 AstrBot 惯例安装依赖即可。

---

## 二、ComfyUI 端依赖（节点与插件）

本插件**只替换**工作流中以下三类节点的输入，其余节点不修改。设计工作流时请仅使用这些节点作为「可注入参数」的入口。

| 用途 | 节点 `class_type` | 输入键名 | 说明 | ComfyUI 插件来源 |
|------|--------------------|----------|------|------------------|
| **文本** | `Simple String` | `text` 或 `string` | 按顺序注入 `texts` 数组 | **[CG Use Everywhere](https://github.com/chrisgoringe/cg-use-everywhere)**（chrisgoringe） |
| **图片** | `ETN_LoadImageBase64` | `image` | 按顺序注入 base64 字符串（PNG） | **ComfyUI Nodes for External Tooling**（如 [comfyui-tooling-nodes](https://comfyai.run/documentation/ETN_LoadImageBase64) / Acly 等），界面显示为 "Load Image (Base64)" |
| **视频** | `VHS_LoadVideo` | `video` | 按顺序注入服务器上的视频文件名（如 .mp4） | **ComfyUI-VideoHelperSuite**（[Kosinkadink/ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)），界面为 "Load Video" 等 |

- **Simple String**：来自 [CG Use Everywhere](https://github.com/chrisgoringe/cg-use-everywhere)，需安装该扩展。
- **ETN_LoadImageBase64**：非内置，需安装 External Tooling / Base64 相关自定义节点，用于接收 base64 图参与推理。
- **VHS_LoadVideo**：来自 VideoHelperSuite，需单独安装该扩展；若工作流不涉及视频可不安。

---

## 三、ComfyUI 工作流设计思路

1. **需要传入的图片**  
   - 在工作流中，凡是要由 LLM/用户传入的**图片**，不要用「Load Image（从路径）」等节点，改用 **ETN_LoadImageBase64**（Base64 节点）作为入口，由本插件在运行时按顺序注入 base64 数据。

2. **需要传入的文本**  
   - 凡是要由 LLM/用户传入的**文本**，使用 **Simple String** 节点作为前级：在需要文本的地方（如 CLIP 文本编码、提示词输入等），用 **Simple String** 的输出连过去，本插件会按顺序把 `texts` 数组注入到这些 Simple String 节点。

3. **顺序与数量**  
   - 工作流中第 1 个 Simple String 对应 `texts[0]`，第 2 个对应 `texts[1]`，以此类推；图片、视频同理。节点数量必须与文件名里的「文本N / 图片M / 视频K」一致（见第四节）。

4. **模型与路径**  
   - 工作流内用到的模型（如 ckpt、LoRA）、路径等，需在**运行本插件的 ComfyUI 环境**中存在，否则会报错（如 `value_not_in_list`）。不同机器需各自准备好相同名称或调整工作流中的模型名。

---

## 四、本插件使用思路

### 4.1 安装与配置

- 将本插件目录放入 AstrBot 的 **`data/plugins/`**（或通过插件市场安装）。  
- 在 AstrBot 配置中填写：
  - **server_ip**：ComfyUI 服务地址（如 `127.0.0.1:8188`）。
  - **client_id**：ComfyUI 客户端 ID（可选，按需填写）。  
- 若需工作流管理页：启用 **webui_enabled**，配置 **webui_host** / **webui_port**（如 `http://127.0.0.1:6187`），重载插件后浏览器访问该地址。

### 4.2 上传工作流与起名（最重要：工作流文件名）

1. **在 ComfyUI 中导出工作流**  
   - 在 ComfyUI 中配置好工作流节点并确保可运行后，选择菜单 **文件 → 导出（API）**，会生成一个 `.json` 文件。  
   - 将该 JSON 文件**重命名**为符合规范的名称（见下文），例如：`改图+文本1+图片1.json`。  
   - **注意**：文件名中的「文本N / 图片M / 视频K」必须与工作流内**可替换节点数量**一致——工作流里有几个 Simple String 就写「文本几」，有几个 ETN_LoadImageBase64 就写「图片几」，有几个 VHS_LoadVideo 就写「视频几」。例如工作流里只有 1 个 Simple String 节点，则文件名中必须是「文本1」。

2. **上传并填写说明**  
   - 打开本插件的**工作流管理页**（见 4.1），点击上传，选择重命名后的 `.json` 文件上传。  
   - 在管理页中为该工作流**输入说明文字**（用于 LLM 选择工作流时参考），保存即可。  
   - 也可将 `.json` 直接放到 **`data/plugin_data/astrbot_plugin_comfyui/workflows/`** 目录下，再在管理页中编辑说明。

3. **工作流文件名规范（核心）**  
   文件名决定「LLM 看到的工作流名」和「需要几个文本/图片/视频」，必须严格按下列格式命名：

   - **格式**：**`工作流名+文本N+图片M+视频K.json`**  
     - `工作流名`：任意中文/英文，即 LLM 调用时的 `workflow_name`（如「改图」「文生图1比1」）。  
     - `文本N`：工作流内须有 N 个 Simple String 节点，对应 `texts` 数组长度 N。  
     - `图片M`：工作流内须有 M 个 ETN_LoadImageBase64 节点，对应 M 张图。  
     - `视频K`：工作流内须有 K 个 VHS_LoadVideo 节点；若无视频可省略整段「+视频K」。  
   - **带输出文本的变体**：**`工作流名+图片1=文本1.json`** 等，用于表示输出文本数量，按需使用。

   **文件名示例（多例）：**

   | 文件名 | 含义 |
   |--------|------|
   | `改图+文本1+图片1.json` | 工作流名「改图」，1 个文本 + 1 张图（1 个 Simple String + 1 个 Base64 节点） |
   | `改图25112+文本1+图片2.json` | 工作流名「改图25112」，1 个文本 + 2 张图 |
   | `文生图1比1+文本1.json` | 工作流名「文生图1比1」，仅 1 个文本，无图 |
   | `文生图9比16+文本1.json` | 工作流名「文生图9比16」，仅 1 个文本 |
   | `文生图16比9+文本1.json` | 工作流名「文生图16比9」，仅 1 个文本 |
   | `手办化+图片1.json` | 工作流名「手办化」，仅 1 张图，无文本 |
   | `改视频+视频1+文本1.json` | 工作流名「改视频」，1 个视频 + 1 个文本 |
   | `改图25113+文本1+图片3.json` | 工作流名「改图25113」，1 个文本 + 3 张图 |

4. **填写说明与 text_slots（推荐）**  
   - 在工作流管理页为每个文件填写**说明**（或直接编辑 `data/plugin_data/astrbot_plugin_comfyui/workflow_meta.json`）。  
   - 说明会通过 `comfyui_list_workflows` 返回给 LLM，便于选择合适工作流。  
   - 若有多段文本，可在 `workflow_meta.json` 中增加 **`text_slots`**，为每个工作流指定各文本槽位的含义（与 Simple String 顺序一致），例如：
     ```json
     {
       "descriptions": {
         "改图+文本1+图片1.json": "根据文本修改图片。flux2_klein9B 模型。",
         "改图25112+文本1+图片2.json": "用于将图片1根据图片2和指定的文本要求进行修改。传入的文本须为「根据图2的XX修改图1」之类。"
       },
       "text_slots": {
         "改图+文本1+图片1.json": ["修改说明"],
         "文生图+文本2.json": ["正面提示词", "负面提示词"]
       }
     }
     ```

### 4.3 LLM 使用流程建议

1. 调用 **`comfyui_list_workflows`**：获取可用工作流列表、说明及所需参数（文本/图片/视频数量及 text_slots）。  
2. 按用户意图选择工作流，若需要图但当前消息无图：先调用 **`get_image_from_context`**（需安装 astrbot_plugin_image_url_base64_to_mcp）获取 URL。  
3. 调用 **`comfyui_execute`**：传入 `workflow_name`、`texts`、`image_urls`（可选）、`videos`（可选）；图片可从当前消息自动提取，或通过 `image_urls` 传入 URL/本地路径（仅限插件数据目录内）。  
4. 调用 **`comfyui_query_wait`**：等待任务完成或获取剩余等待时间；结果若为图片，插件会下载后发送到当前会话。  
5. 可选：**`comfyui_status`** 查看队列状态。

### 4.4 安全与目录

- **本地图片路径**：`comfyui_execute` 的 `image_urls` 若传入本地路径，仅允许以下根目录之下：**插件数据目录**（`data/plugin_data/astrbot_plugin_comfyui/`）、**`data/agent/comfyui/input/`**、**`data/temp/`**（平台/适配器存放用户上传图的临时目录，避免「图在 temp 不被认可」导致 images=0）。建议使用绝对路径。禁止 `../` 路径穿越。
- **占位符与持久化路径**：发送 ComfyUI 生成的图片/视频时，插件会将其另存到 `data/agent/comfyui/input/` 并在消息中追加「ComfyUI 图片路径: /path」或「ComfyUI 视频路径: /path」。`qts_get_recent_messages` 等返回的 content 会包含该路径，Bot 可解析后作为下一轮 `image_urls` 传入。
- **清理本地缓存**：工作流管理页提供「清理本地缓存」按钮，可删除 `data/agent/comfyui/input/` 与插件 `tmp/` 下的文件，防止占用过多磁盘空间。  
- **base64 不传入 LLM**：优先使用 URL 或本地路径；若图片来源工具返回占位符（如 `base64://ASTRBOT_PLUGIN_CACHE_PENDING`），请将占位符传入 `image_urls`，不要将原始 base64 填入工具参数，以免 base64 进入 LLM 上下文。插件侧已对相关日志脱敏。  
- 插件数据目录：**`data/plugin_data/astrbot_plugin_comfyui/`**，其中 **`workflows/`** 存放工作流 JSON，**`workflow_meta.json`** 存放说明与 text_slots。

---

## 五、功能概览（简要）

| 能力 | 说明 |
|------|------|
| **comfyui_list_workflows** | 查询可用工作流及说明、所需文本/图片/视频数量及 text_slots |
| **comfyui_execute** | 提交工作流；支持 texts、image_urls（URL 或允许范围内的本地路径）、videos |
| **comfyui_query_wait** | 等待任务完成或返回剩余时间；若结果为图片则下载并发送到会话 |
| **comfyui_status** | 查询 ComfyUI 队列状态（运行中/等待中数量） |
| **工作流管理页** | 上传/重命名/删除工作流 JSON，编辑说明；可选启用 |

---

## 六、注意事项小结

- 工作流中**仅** Simple String、ETN_LoadImageBase64、VHS_LoadVideo 会被本插件替换；其他节点请勿依赖「由本插件注入」的文本/图/视频。  
- 工作流文件名必须符合 **`工作流名+文本N+图片M+视频K.json`** 规范，否则无法正确匹配参数数量。  
- 推荐安装 [astrbot_plugin_image_url_base64_to_mcp](https://github.com/Thetail001/astrbot_plugin_image_url_base64_to_mcp)，以便在「用户引用之前消息的图」时通过 `get_image_from_context` + `image_urls` 完成改图。

---

