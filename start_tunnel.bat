@echo off
chcp 65001 >nul
title MIMII Tunnel

echo ============================================
echo   MIMII Acoustic Diagnostics
echo ============================================
echo.
echo Starting Flask server...
start /B pythonw gui.py
timeout /t 3 /nobreak >nul

echo Starting Cloudflare Tunnel...
echo The URL will appear in 5-10 seconds.
echo Open it on your phone to access the interface.
echo Close this window to stop everything.
echo.

"C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://localhost:8080 --no-autoupdate

echo.
echo Tunnel stopped.
pause
