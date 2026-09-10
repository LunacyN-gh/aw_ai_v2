@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m aw_ai gui
    if errorlevel 1 pause
    exit /b
)
where python >nul 2>nul
if %errorlevel%==0 (
    python -m aw_ai gui
) else (
    if exist "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" (
        "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m aw_ai gui
    ) else (
        py -3 -m aw_ai gui
    )
)
if errorlevel 1 pause
