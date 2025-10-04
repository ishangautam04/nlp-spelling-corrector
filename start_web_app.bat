@echo off
echo Starting NLP Spelling Corrector Web Interface...
echo.
cd /d "C:\Users\ishan\Desktop\spelling corrector"
call spelling_corrector_env\Scripts\activate
echo Virtual environment activated.
echo.
echo Starting Streamlit app...
spelling_corrector_env\Scripts\streamlit.exe run streamlit_app.py --server.port 8504
echo.
echo App has stopped.
pause
