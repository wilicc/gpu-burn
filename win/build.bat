@echo off
REM Build script for Windows GPU Burn.

setlocal enabledelayedexpansion

set COMPUTE=75
set BUILD_TYPE=Release

:parse_args
if "%~1"=="" goto :build
if "%~1"=="-h" goto :help
if "%~1"=="--help" goto :help
if "%~1"=="-c" (
    set COMPUTE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="-d" (
    set BUILD_TYPE=Debug
    shift
    goto :parse_args
)
echo Unknown argument: %~1
goto :help

:build
where cmake >nul 2>&1 || (echo ERROR: cmake not in PATH & exit /b 1)
where nvcc  >nul 2>&1 || echo WARNING: nvcc not in PATH

if not exist build mkdir build
cd build

cmake .. -DCOMPUTE=%COMPUTE% -DCMAKE_BUILD_TYPE=%BUILD_TYPE% || (cd .. & exit /b 1)
cmake --build . --config %BUILD_TYPE%                       || (cd .. & exit /b 1)

cd ..
echo.
echo Built %BUILD_TYPE% gpu_burn.exe for compute_%COMPUTE% in build\%BUILD_TYPE%\
goto :end

:help
echo Usage: build.bat [-c COMPUTE] [-d]
echo.
echo   -c VALUE   CUDA compute capability (default: 75)
echo   -d         Debug build (default: Release)
echo.
echo Example: build.bat -c 86

:end
endlocal
