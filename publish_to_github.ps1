# Usage: .\publish_to_github.ps1 -RepoUrl "https://github.com/cjxzdzh/astrbot_plugin_comfyui.git"
param([Parameter(Mandatory=$true)][string]$RepoUrl)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".git")) { git init }
git add -A
$out = git status --short 2>$null
if (-not $out) { Write-Host "No changes to commit."; exit 0 }
git commit -m "chore: initial commit - AstrBot ComfyUI workflow plugin (LLM tools)"
git branch -M main 2>$null
if (git remote get-url origin 2>$null) { git remote remove origin }
git remote add origin $RepoUrl
git push -u origin main
Write-Host "Done. Set repo to Public in GitHub Settings if needed."
