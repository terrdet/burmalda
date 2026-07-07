@echo off
cd /d "%~dp0"
echo Stopping server...
wmic process where "name='python.exe' and commandline like '%%gui.py%%'" call terminate >nul 2>&1
wmic process where "name='py.exe' and commandline like '%%gui.py%%'" call terminate >nul 2>&1
wmic process where "name='wscript.exe' and commandline like '%%start_gui.vbs%%'" call terminate >nul 2>&1
timeout /t 1 /nobreak >nul
echo Starting...
start /b wscript.exe start_gui.vbs
echo Done.
