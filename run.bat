@echo off
title UE5 Stress Simulator
cls

echo ========================================================
echo                       PYBECRASHER
echo    UE5 SHADER COMPILATION AND OODLE STRESS SIMULATOR
echo ========================================================
echo.
echo This tool simulates the Unreal Engine 5 "Preparing Shaders"
echo workload plus Oodle decompression to test CPU/RAM stability
echo.
echo Select Mode:
echo.
echo [1] Variable "Chaos" Load
echo     - Cycles between Single Core, Ramp Up, Random, and
echo       Transient Spikes (Idle to Max instantly).
echo.
echo [2] Steady Load
echo     - Constant 100%% load.
echo     - Best for thermal testing (and performance comparison).
echo.

set "choice=1"
set /p choice="Enter selection (default is 1): "

set MODE_ARG=--mode variable
if "%choice%"=="2" set MODE_ARG=--mode steady

echo.
echo Starting Stress Test in %MODE_ARG% mode...
echo Press Ctrl+C to stop at any time.
echo.

:: Run the Python script
python ue5_shader_stress.py %MODE_ARG%

:: Keep window open
echo.
echo ========================================================
echo Test execution finished (or crashed).
echo ========================================================
pause