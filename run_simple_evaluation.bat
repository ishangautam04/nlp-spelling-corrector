@echo off
REM Simple Test Evaluation - No Training Required
REM Tests all 4 algorithms on entire large dataset

echo ========================================
echo  Spelling Correction Test Evaluation
echo ========================================
echo.
echo Testing all 4 pre-trained algorithms
echo No training required!
echo.

REM Activate virtual environment
call .venv\Scripts\activate

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate virtual environment!
    pause
    exit /b 1
)

echo Virtual environment activated
echo.

REM Check if dataset exists
if not exist "spelling_test_dataset_large.json" (
    echo ERROR: Dataset not found: spelling_test_dataset_large.json
    echo Please make sure the dataset file exists.
    pause
    exit /b 1
)

echo Starting evaluation...
echo This will test on all samples in the large dataset
echo.
echo ========================================
echo.

REM Run evaluation
python simple_test_evaluation.py --dataset spelling_test_dataset_large.json --output evaluation_results.json

echo.
echo ========================================
echo.
echo Evaluation complete!
echo.
echo Results saved to:
echo   - evaluation_results.json  (detailed results)
echo   - evaluation_results.csv   (summary table)
echo.
pause
