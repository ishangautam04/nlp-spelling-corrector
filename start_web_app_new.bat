@echo off
REM NLP Spelling Corrector - Algorithm Comparison Web App
REM Updated version without ensemble approach

echo ========================================
echo  Spelling Correction Comparison Tool
echo ========================================
echo.
echo Starting web application...
echo.

REM Activate virtual environment
call .venv\Scripts\activate

REM Check if activation was successful
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate virtual environment!
    echo Please ensure .venv folder exists and contains a valid Python environment.
    pause
    exit /b 1
)

echo Virtual environment activated
echo.
echo Starting comparison-focused app...
echo.

REM Run the streamlit app
streamlit run streamlit_app.py --server.port 8504

echo.
echo App has stopped.
pause
