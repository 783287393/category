@echo off
rem 一键启动 Wine 智能分类系统（Windows 批处理入口）
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
    echo 未找到 .venv，请先执行: python -m venv .venv 并 pip install -r requirements.txt
    pause
    exit /b 1
)
if not exist "models\model.joblib" (
    echo 未找到模型文件，请先执行: .\.venv\Scripts\python.exe -m src.train
    pause
    exit /b 1
)
echo 启动 Wine 智能分类系统: http://localhost:8501
.\.venv\Scripts\streamlit.exe run app.py
