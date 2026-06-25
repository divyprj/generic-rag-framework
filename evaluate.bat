@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

rem Usage: evaluate.bat [dataset] [provider] [--ingest]
set "DATASET=%~1"
set "PROVIDER=%~2"
set "EXTRA_ARGS="

if "%DATASET%"=="" set "DATASET=quantum"
if "%PROVIDER%"=="" set "PROVIDER=groq"

for %%a in (%*) do (
    if /i "%%a"=="--ingest" set "EXTRA_ARGS=!EXTRA_ARGS! --ingest"
    if /i "%%a"=="-i" set "EXTRA_ARGS=!EXTRA_ARGS! --ingest"
)

echo.
echo ============================================================
echo    GENERIC RAG FRAMEWORK - EVALUATION
echo    Dataset  : %DATASET%
echo    Provider : %PROVIDER%
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found.
    echo         Please run install.bat first.
    goto :fail
)
set "PYTHON=.venv\Scripts\python.exe"
echo [OK] Virtual environment found.
echo [OK] Using virtual environment Python.

if not exist ".\\.env" (
    echo [ERROR] .env file not found.
    echo         Create .env with your API keys.
    echo         Or run: copy .env.example .env
    goto :fail
)

if not exist "data\%DATASET%\qa_dataset.json" (
    echo [ERROR] QA dataset not found: data\%DATASET%\qa_dataset.json
    echo.
    echo         Available datasets:
    for /d %%D in (data\*) do (
        if exist "%%D\documents" echo           %%~nD
    )
    echo.
    echo         Usage: evaluate.bat ^<dataset^> ^<provider^>
    goto :fail
)
echo [OK] QA dataset found.

dir /b "data\%DATASET%\documents\*.md" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] No .md documents found in data\%DATASET%\documents\
    goto :fail
)
echo [OK] Documents found.
echo.

echo [INFO] Evaluating dataset=%DATASET% provider=%PROVIDER%
echo.
echo        The evaluation will:
echo          1. Load and index documents
echo          2. Run all questions through the RAG pipeline
echo          3. Compute quantitative metrics
echo          4. Compute retrieval metrics (P@k, R@k, MRR)
echo          5. Generate reports in results\
echo.
echo ============================================================
echo.

"%PYTHON%" -m evaluation.run_evaluation --dataset "%DATASET%" --provider "%PROVIDER%" !EXTRA_ARGS!
if errorlevel 1 (
    echo.
    echo [ERROR] Evaluation failed. Check the errors above.
    goto :fail
)

echo.
echo ============================================================
echo    EVALUATION COMPLETE
echo ============================================================
echo.

echo    Generated Reports:
echo.
if exist "results\evaluation.json" echo      - results\evaluation.json    Full structured results
if exist "results\evaluation.csv" echo      - results\evaluation.csv     Spreadsheet-friendly
if exist "results\evaluation.md" echo      - results\evaluation.md      Formatted report
echo.
echo ============================================================
echo.
if not defined NO_PAUSE pause
exit /b 0

:fail
echo.
echo ============================================================
echo    EVALUATION FAILED
echo ============================================================
echo    Please fix the issue above and try again.
echo ============================================================
echo.
if not defined NO_PAUSE pause
exit /b 1
