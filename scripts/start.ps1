# 启动 Streamlit 推理应用
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot)

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "未找到 .venv，请先执行: python -m venv .venv 并 pip install -r requirements.txt"
    exit 1
}
if (-not (Test-Path "models\model.joblib")) {
    Write-Host "未找到模型文件，请先执行: .\.venv\Scripts\python.exe -m src.train"
    exit 1
}

Write-Host "启动 Wine 智能分类系统: http://localhost:8501"
& ".\.venv\Scripts\streamlit.exe" run app.py
