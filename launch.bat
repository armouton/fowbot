@echo off
echo ====================================
echo  Next-Word Predictor - Web Version
echo ====================================
echo.

REM Try to find a working Python installation
set PYTHON=

REM 1. Check if 'python' actually works (not just the Windows Store stub)
python --version >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set PYTHON=python
    goto :found
)

REM 2. Check python3
python3 --version >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set PYTHON=python3
    goto :found
)

REM 3. Check common Anaconda/Miniconda locations
if exist "%LOCALAPPDATA%\anaconda3\python.exe" (
    set "PYTHON=%LOCALAPPDATA%\anaconda3\python.exe"
    goto :found
)
if exist "%LOCALAPPDATA%\miniconda3\python.exe" (
    set "PYTHON=%LOCALAPPDATA%\miniconda3\python.exe"
    goto :found
)
if exist "%USERPROFILE%\anaconda3\python.exe" (
    set "PYTHON=%USERPROFILE%\anaconda3\python.exe"
    goto :found
)
if exist "%USERPROFILE%\miniconda3\python.exe" (
    set "PYTHON=%USERPROFILE%\miniconda3\python.exe"
    goto :found
)
if exist "%ProgramData%\anaconda3\python.exe" (
    set "PYTHON=%ProgramData%\anaconda3\python.exe"
    goto :found
)

REM 4. Check standard Python install locations
for %%V in (313 312 311 310 39) do (
    if exist "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe" (
        set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe"
        goto :found
    )
)

REM 5. Try conda activate as a last resort
where conda >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo Found Conda. Activating base environment...
    call conda activate base
    set PYTHON=python
    goto :found
)

echo.
echo Python not found!
echo.
echo Please install Python 3.8+ using one of these options:
echo   - Download from https://www.python.org (check "Add Python to PATH")
echo   - Or install Anaconda from https://www.anaconda.com
echo.
echo If you have Anaconda installed, try launching from the
echo Anaconda Prompt instead of double-clicking this file.
echo.
pause
exit /b 1

:found
echo Found Python:
"%PYTHON%" --version
echo.
echo This will install required packages (numpy, flask) if not already present.
echo Press any key to continue, or close this window to cancel.
pause >nul
echo.

REM Install dependencies
echo Installing dependencies (if needed)...
"%PYTHON%" -m pip install -r requirements.txt --quiet
echo.

REM Launch server and open browser
echo Starting server at http://localhost:5001
echo Close this window to stop the server.
echo.
start http://localhost:5001
"%PYTHON%" app.py
