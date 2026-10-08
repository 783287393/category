# 停止 Streamlit 推理应用
$ErrorActionPreference = "SilentlyContinue"
$proc = Get-Process -Name streamlit -ErrorAction SilentlyContinue
if ($proc) {
    $proc | Stop-Process -Force
    Write-Host "Streamlit 已停止。"
} else {
    Write-Host "未发现运行中的 Streamlit 进程。"
}
