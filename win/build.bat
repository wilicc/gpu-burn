@echo off
REM Build script for Windows GPU Burn
REM Supports multiple compute capabilities

setlocal enabledelayedexpansion

REM Default values
set COMPUTE=75
set BUILD_TYPE=Release

REM Parse arguments
:parse_args
if "%1"=="" goto :build
if "%1"=="-h" goto :help
if "%1"=="--help" goto :help
if "%1"=="-c" (
    set COMPUTE=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--compute" (
    set COMPUTE=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="COMPUTE=" (
    set COMPUTE=%1
    set COMPUTE=!COMPUTE:COMPUTE=!
    shift
    goto :parse_args
)
if "%1"=="-d" (
    set BUILD_TYPE=Debug
    shift
    goto :parse_args
)
if "%1"=="--debug" (
    set BUILD_TYPE=Debug
    shift
    goto :parse_args
)
shift
goto :parse_args

:build
echo Building GPU Burn for Windows
echo Compute Capability: %COMPUTE%
echo Build Type: %BUILD_TYPE%

REM Check if CMake is available
where cmake >nul 2>&1
if errorlevel 1 (
    echo ERROR: CMake is not found in PATH
    echo Please install CMake and add it to PATH
    exit /b 1
)

REM Check if CUDA is available
where nvcc >nul 2>&1
if errorlevel 1 (
    echo WARNING: nvcc is not found in PATH
    echo Make sure CUDA Toolkit is installed
)

REM Create build directory
if not exist build mkdir build
cd build

REM Configure with CMake
cmake .. -DCOMPUTE=%COMPUTE% -DCMAKE_BUILD_TYPE=%BUILD_TYPE%
if errorlevel 1 (
    echo ERROR: CMake configuration failed
    cd ..
    exit /b 1
)

REM Build
cmake --build . --config %BUILD_TYPE%
if errorlevel 1 (
    echo ERROR: Build failed
    cd ..
    exit /b 1
)

cd ..
echo.
echo Build completed successfully!
if "%BUILD_TYPE%"=="Release" (
    echo Executable: build\Release\gpu_burn.exe
    echo Fatbin file: build\Release\compare.fatbin
) else (
    echo Executable: build\Debug\gpu_burn.exe
    echo Fatbin file: build\Debug\compare.fatbin
)
goto :end

:help
echo Usage: build.bat [OPTIONS]
echo.
echo Options:
echo   -c, --compute VALUE    Set compute capability (default: 75)
echo                          Supported: 75, 80, 86, 89, 90, 100, 120
echo   COMPUTE=VALUE          Set compute capability (Linux style)
echo                          Supported: 75, 80, 86, 89, 90, 100, 120
echo   -d, --debug           Build in Debug mode (default: Release)
echo   -h, --help           Show this help message
echo.
echo Examples:
echo   build.bat                    Build with default compute 75 (Turing)
echo   build.bat -c 80              Build for compute 80 (Ampere)
echo   build.bat -c 86              Build for compute 86 (Ampere)
echo   build.bat -c 89              Build for compute 89 (Ada)
echo   build.bat -c 90              Build for compute 90 (Ada)
echo   build.bat -c 100             Build for compute 100 (Hopper)
echo   build.bat -c 120             Build for compute 120 (Blackwell)
echo   build.bat COMPUTE=120        Build for compute 120 (Linux style)
echo   build.bat COMPUTE=86 -d      Build for compute 86 in Debug mode
goto :end

:end
endlocal

