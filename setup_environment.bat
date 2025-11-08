@echo off
REM Quick Setup Script for Data Generation Project (Windows)
REM Double-click this file to set up everything automatically

echo 🚀 Setting up Data Generation Project...
echo ==========================================

REM Check if we're in the right directory
if not exist "setup.py" (
    echo ❌ Error: setup.py not found!
    echo    Please run this script from the project root directory
    pause
    exit /b 1
)

echo ✅ Found project root

REM Create virtual environment
echo.
echo 📦 Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo 🔌 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo.
echo 📥 Installing dependencies (this may take a few minutes)...
pip install -r requirements.txt

REM Install project as package
echo.
echo 📦 Installing project as editable package...
pip install -e .

REM Install Jupyter
echo.
echo 📓 Installing Jupyter...
pip install jupyter ipywidgets ipykernel

echo.
echo ==========================================
echo ✅ Setup complete!
echo.
echo 🎯 Next steps:
echo    1. Activate the environment:
echo       venv\Scripts\activate
echo    2. Start Jupyter:
echo       jupyter notebook
echo    3. Open: notebooks/01_getting_started.ipynb
echo.
echo 🎉 Happy generating!
echo.
pause

