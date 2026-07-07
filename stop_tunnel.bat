@echo off
echo Stopping MIMII Tunnel and Flask...
PowerShell -Command "Get-CimInstance Win32_Process -Filter \"Name='cloudflared.exe'\" | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
PowerShell -Command "Get-CimInstance Win32_Process -Filter \"Name='pythonw.exe' OR Name='python.exe'\" | Where-Object { $_.CommandLine -match 'gui.py' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
echo Stopped.
pause
