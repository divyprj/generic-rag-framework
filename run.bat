@echo off
setlocal
cd /d "%~dp0"

rem Usage: run.bat [dataset] [provider] [--fallback]
set "DATASET=%~1"
set "PROVIDER=%~2"
set "EXTRA_ARGS="

if "%DATASET%"=="" set "DATASET=quantum"
if "%PROVIDER%"=="" set "PROVIDER=groq"

for %%A in (%*) do (
    if /I "%%~A"=="--fallback" set "EXTRA_ARGS=--fallback"
)

echo.
echo ============================================================
echo    GENERIC RAG FRAMEWORK
echo    Dataset  : %DATASET%
echo    Provider : %PROVIDER%
echo ============================================================
echo.

if not exist "data\%DATASET%\documents" (
    echo [ERROR] Dataset "%DATASET%" not found.
    echo         Expected folder: data\%DATASET%\documents\
    echo.
    echo         Available datasets:
    for /d %%D in (data\*) do (
        if exist "%%D\documents" echo           %%~nD
    )
    echo.
    echo         Usage: run.bat ^<dataset^> ^<provider^> [--fallback]
    echo         Example: run.bat mars groq
    goto :fail
)

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found.
    echo         Please run install.bat first.
    goto :fail
)
set "PYTHON=.venv\Scripts\python.exe"
echo [OK] Virtual environment found.

if not exist ".\\.env" (
    echo.
    echo [ERROR] .env file not found.
    echo.
    echo         To fix this:
    echo           1. Copy the template: copy .env.example .env
    echo           2. Edit .env and add your API keys.
    echo.
    echo         Get API keys at:
    echo           Groq:   https://console.groq.com/keys
    echo           Gemini: https://aistudio.google.com/apikey
    echo.
    goto :fail
)
echo [OK] .env file found.
echo.

echo [INFO] Starting with dataset=%DATASET% provider=%PROVIDER%
echo        Type help for commands, quit to exit.
echo.
echo ============================================================
echo.
"%PYTHON%" app.py --dataset "%DATASET%" --provider "%PROVIDER%" %EXTRA_ARGS%
if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with an error.
    echo         Check the error messages above.
    goto :fail
)

echo.
echo ============================================================
echo    Application closed.
echo ============================================================
echo.
if not defined NO_PAUSE pause
exit /b 0

:fail
echo.
echo ============================================================
echo    Please fix the issue above and try again.
echo ============================================================
echo.
if not defined NO_PAUSE pause
exit /b 1
