@echo off
REM Launcher for Spelling Correction Apps
REM Allows user to choose between original and comparison-focused app

:MENU
cls
echo ========================================
echo  NLP Spelling Corrector - Launcher
echo ========================================
echo.
echo Please select which app to run:
echo.
echo [1] NEW: Algorithm Comparison Tool
echo     (Side-by-side comparison, no ensemble)
echo.
echo [2] ORIGINAL: Full-featured App
echo     (Includes ensemble, ML model, all features)
echo.
echo [3] Exit
echo.
echo ========================================
echo.

set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" goto NEWCOMPARISON
if "%choice%"=="2" goto ORIGINAL
if "%choice%"=="3" goto EXIT

echo Invalid choice! Please enter 1, 2, or 3.
timeout /t 2 >nul
goto MENU

:NEWCOMPARISON
echo.
echo Starting Algorithm Comparison Tool...
echo.
call .venv\Scripts\activate
if exist "streamlit_app.py" (
    streamlit run streamlit_app.py --server.port 8504
) else (
    echo ERROR: streamlit_app.py not found!
    pause
)
goto END

:ORIGINAL
echo.
echo Starting ORIGINAL Full-featured App (Backup)...
echo.
call .venv\Scripts\activate
if exist "streamlit_app_old_backup.py" (
    streamlit run streamlit_app_old_backup.py --server.port 8505
) else (
    echo ERROR: streamlit_app_old_backup.py not found!
    pause
)
goto END

:EXIT
echo.
echo Goodbye!
timeout /t 1 >nul
exit

:END
echo.
echo App has stopped.
echo.
pause
goto MENU
