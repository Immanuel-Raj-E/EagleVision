@echo off
cd /d "%~dp0"
set PYTHONPATH=%~dp0;%PYTHONPATH%
call d:\SEC\yolo_env\Scripts\activate
streamlit run ui/app.py
