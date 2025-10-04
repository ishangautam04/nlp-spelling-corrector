@echo off
echo Starting NLP Spelling Corrector CLI...
echo.
cd /d "C:\Users\ishan\Desktop\spelling corrector"
call spelling_corrector_env\Scripts\activate
echo Virtual environment activated.
echo.
echo Starting interactive CLI mode...
python cli.py --interactive
echo.
echo CLI has stopped.
pause
