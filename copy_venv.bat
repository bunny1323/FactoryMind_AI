@echo off
echo Copying virtual environment from downloads to workspace...
xcopy /E /I /Y "c:\Users\vigne\Downloads\medmitra_fastapi_qdrant_prod\venv" "c:\Users\vigne\Desktop\FactoryMind AI\venv"
echo Virtual environment copy complete!
pause
