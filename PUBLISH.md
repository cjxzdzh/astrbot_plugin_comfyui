# 发布到 GitHub（公开仓库）

## 隐私检查说明

发布前已确认本仓库中**未包含**：

- 账号、密码、API Key、Token 等敏感信息
- 包含个人姓名的路径（如用户目录）
- 本机或内网专用地址（仅保留示例用 `127.0.0.1`、`192.168.x.x` 等说明）

可放心公开。

---

## 一键发布步骤

1. **在 GitHub 上新建空仓库**
   - 打开 https://github.com/new
   - 仓库名建议：`astrbot_plugin_comfyui`
   - 选择 **Public**
   - **不要**勾选 “Add a README file” / “Add .gitignore”
   - 创建仓库后，复制仓库的 HTTPS 地址，例如：`https://github.com/你的用户名/astrbot_plugin_comfyui.git`

2. **在本插件目录执行脚本**
   - 在 PowerShell 中进入本目录，执行：
   ```powershell
   cd "插件所在路径\astrbot_plugin_comfyui"
   .\publish_to_github.ps1 -RepoUrl "https://github.com/你的用户名/astrbot_plugin_comfyui.git"
   ```
   - 按提示输入 GitHub 账号密码或使用已配置的凭据（如 SSH/Token）。

3. **确认仓库为公开**
   - 打开仓库 → **Settings** → **General** → **Danger Zone** 上方可见可见性为 **Public**；若为 Private，可在此改为 Public。

---

## 手动执行（不用脚本时）

```powershell
cd "插件所在路径\astrbot_plugin_comfyui"
git init
git add -A
git commit -m "chore: initial commit - AstrBot ComfyUI workflow plugin (LLM tools)"
git branch -M main
git remote add origin https://github.com/你的用户名/astrbot_plugin_comfyui.git
git push -u origin main
```

完成上述任一步骤后，项目即已上传到 GitHub 并可为公开仓库。
