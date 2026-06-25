@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

:: ── Enable ANSI Colors ───────────────────────────────────────
:: ESC character for ANSI codes (works on Windows 10+)
for /f %%a in ('echo prompt $E ^| cmd') do set "ESC=%%a"
set "GREEN=%ESC%[92m"
set "RED=%ESC%[91m"
set "YELLOW=%ESC%[93m"
set "CYAN=%ESC%[96m"
set "BOLD=%ESC%[1m"
set "RESET=%ESC%[0m"

echo.
echo %CYAN%============================================================%RESET%
echo %BOLD%   MARS EXPLORATION RAG SYSTEM - INSTALLER%RESET%
echo %CYAN%============================================================%RESET%
echo.

:: ── Check Python ─────────────────────────────────────────────
echo %CYAN%[INFO]%RESET% Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo %RED%[ERROR]%RESET% Python is not installed or not in PATH.
    echo         Please install Python 3.10+ from https://www.python.org/downloads/
    echo         Make sure to check "Add Python to PATH" during installation.
    goto :fail
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo %GREEN%[SUCCESS]%RESET% Python %BOLD%%PYVER%%RESET% found.
echo.

:: ── Create Virtual Environment ───────────────────────────────
if exist ".venv\Scripts\python.exe" (
    echo %CYAN%[INFO]%RESET% Virtual environment already exists. Skipping creation.
) else (
    echo %CYAN%[INFO]%RESET% Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo %RED%[ERROR]%RESET% Failed to create virtual environment.
        goto :fail
    )
    echo %GREEN%[SUCCESS]%RESET% Virtual environment created.
)
echo.

:: ── Activate Virtual Environment ─────────────────────────────
echo %CYAN%[INFO]%RESET% Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo %RED%[ERROR]%RESET% Failed to activate virtual environment.
    goto :fail
)
echo %GREEN%[SUCCESS]%RESET% Virtual environment activated.
echo.

:: ── Upgrade pip ──────────────────────────────────────────────
echo %CYAN%[INFO]%RESET% Upgrading pip...
python -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo %YELLOW%[WARN]%RESET% Could not upgrade pip. Continuing anyway...
) else (
    echo %GREEN%[SUCCESS]%RESET% pip upgraded.
)
echo.

:: ── Install Dependencies ─────────────────────────────────────
echo %CYAN%[INFO]%RESET% Installing dependencies from requirements.txt...
echo         This may take several minutes on first install
echo         (PyTorch + sentence-transformers are large packages).
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo %RED%[ERROR]%RESET% Dependency installation failed.
    echo         Check the error messages above for details.
    goto :fail
)
echo.

:: ── Verify Installation ─────────────────────────────────────
echo %CYAN%[INFO]%RESET% Verifying installation...
python -c "from src.config import *; from src.document_loader import load_documents; from src.embeddings import EmbeddingModel; from src.vector_store import VectorStore; from src.retriever import Retriever; from src.generator import Generator; from src.rag_pipeline import RAGPipeline; from evaluation.metrics import compute_all_metrics; from evaluation.retrieval_eval import evaluate_retrieval"
if errorlevel 1 (
    echo.
    echo %RED%[ERROR]%RESET% Module verification failed.
    echo         Some dependencies may not have installed correctly.
    goto :fail
)
echo %GREEN%[SUCCESS]%RESET% All modules verified.
echo.

:: ── Success ──────────────────────────────────────────────────
echo %GREEN%============================================================%RESET%
echo %BOLD%   INSTALLATION COMPLETE%RESET%
echo %GREEN%============================================================%RESET%
echo.
echo    Next steps:
echo.
echo      %BOLD%1.%RESET% Create a .env file with your Gemini API key:
echo         %CYAN%GOOGLE_API_KEY=your_key_here%RESET%
echo.
echo      %BOLD%2.%RESET% Run the application:
echo         %CYAN%run.bat%RESET%
echo.
echo      %BOLD%3.%RESET% Run the evaluation:
echo         %CYAN%evaluate.bat%RESET%
echo.
echo    Get a free API key at:
echo      %CYAN%https://aistudio.google.com/apikey%RESET%
echo.
echo %GREEN%============================================================%RESET%
echo.
pause
exit /b 0

:fail
echo.
echo %RED%============================================================%RESET%
echo %BOLD%   INSTALLATION FAILED%RESET%
echo %RED%============================================================%RESET%
echo    Please review the error messages above.
echo %RED%============================================================%RESET%
echo.
pause
exit /b 1
