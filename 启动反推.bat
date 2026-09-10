@echo off
cd /d "%~dp0"
if exist "%~dp0start-local.ps1" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-local.ps1"
  if errorlevel 1 pause
  exit /b %errorlevel%
)
powershell -NoProfile -Command "try { $c = New-Object Net.Sockets.TcpClient('127.0.0.1',18770); $c.Close(); exit 0 } catch { exit 1 }"
if %errorlevel%==0 (
  start "" http://127.0.0.1:18770/
  exit /b 0
)
python -m nai5_tagger.server
pause
