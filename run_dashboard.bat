@echo off
set PYTHONPATH=%cd%
echo Starting Causal Air Apportionment Dashboard...

:: Use the explicit Anaconda path to avoid global PATH issues
"C:\Users\Fasihuddin\anaconda3\Scripts\streamlit.exe" run app/dashboard.py

:: Pause keeps the window open so you can see any error messages
pause
