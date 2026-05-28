# GPU Burn Windows Port - Development Context

## Project Overview

This document records the development process of porting GPU Burn from Linux to Windows, and adding support for new GPU architectures (sm_100 and sm_120).

## Date: 2025-11-08

## Original Project

- **Source**: Linux GPU Burn tool (https://github.com/wilicc/gpu-burn)
- **Purpose**: Multi-GPU CUDA stress testing tool
- **License**: BSD-style license

## Porting Requirements

1. Port Linux-specific code to Windows
2. Maintain feature parity with Linux version
3. Add support for compute capability 10.0 (Hopper - sm_100)
4. Add support for compute capability 12.0 (Blackwell - sm_120)
5. Create Windows build system
6. Compile to executable (.exe)

## Key Changes Made

### 1. Process Management

**Linux (Original)**:
- Used `fork()` to create child processes
- Used Unix pipes (`pipe()`) for inter-process communication
- Used `waitpid()` for process management
- Used `kill()` with SIGTERM/SIGKILL for termination

**Windows (Port)**:
- Replaced with Windows threads (`_beginthreadex`)
- Used Windows named pipes (`CreatePipe`) for IPC
- Used `WaitForMultipleObjects` for thread synchronization
- Used `TerminateThread`/`TerminateProcess` for termination

### 2. Signal Handling

**Linux**:
- Used `sigaction()` for SIGTERM handling
- Global flag `g_running` controlled by signal handler

**Windows**:
- Removed signal handling (not needed with threads)
- Direct flag manipulation for thread control

### 3. Time Functions

**Linux**:
- `gettimeofday()` for time measurement
- `time()` for seconds since epoch
- `clock_gettime()` for high-resolution time

**Windows**:
- `GetSystemTimeAsFileTime()` for time measurement
- `time()` still available (C runtime)
- `FILETIME` structure for high-resolution time

### 4. Temperature Monitoring

**Linux**:
- Forked process running `nvidia-smi` or `tegrastats`
- Read from pipe

**Windows**:
- `CreateProcess()` to launch `nvidia-smi.exe`
- Read from pipe handle
- Similar parsing logic

### 5. Build System

**Linux**:
- Makefile with g++ and nvcc
- Default compute capability: 7.5

**Windows**:
- CMake-based build system
- Support for multiple compute capabilities
- Batch scripts for easy building
- Visual Studio compatible

### 6. Compute Capability Support

**Original Support**:
- Default: 7.5 (Turing, Ampere)

**Added Support**:
- 10.0 (sm_100) - Hopper architecture (H100, etc.)
- 12.0 (sm_120) - Blackwell architecture (B100, etc.)

These are specified via CMake `-DCOMPUTE` flag or build.bat `-c` option.

## File Structure

```
win/
├── gpu_burn-drv.cpp    # Main Windows port (replaces fork with threads)
├── compare.cu          # CUDA kernel (unchanged from original)
├── CMakeLists.txt      # CMake build configuration
├── build.bat           # Build script for single compute capability
├── README.md           # Usage documentation
└── context.md          # This file
```

## Technical Details

### Thread Model

Instead of forking processes, Windows version uses threads:
- Each GPU gets its own thread
- Threads communicate via pipes
- Main thread monitors all GPU threads and temperature process

### Memory Management

- Same memory allocation strategy as Linux version
- Uses 90% of available GPU memory by default
- Supports percentage-based and absolute memory limits

### Error Handling

- Maintains same error checking and reporting
- Uses CUDA error codes
- Thread-safe error accumulation

### CUDA API Usage

- Uses CUDA Driver API (cuInit, cuDeviceGet, etc.)
- Uses CUBLAS for matrix operations
- Fatbin kernel loading for comparison operations

## Build Instructions

### Prerequisites
1. CUDA Toolkit 11.0+ (13.0 recommended for compute 12.0)
2. CMake 3.15+
3. Visual Studio 2019+ with C++ tools
4. Windows 10/11

### Build Commands

```batch
# Default (sm_75, compute 7.5)
build.bat

# Hopper (sm_100, compute 10.0)
build.bat -c 100

# Blackwell (sm_120, compute 12.0)
build.bat -c 120


| GPU Arch | Compute Capability | CUDA 11   | CUDA 12   | CUDA 13  |
|----------|--------------------|-----------|-----------|----------|
| sm_75    | 7.5                | ✅        | ✅        | ✅       |
| sm_80    | 8.0                | ✅        | ✅        | ✅       |
| sm_86    | 8.6                | ✅        | ✅        | ✅       |
| sm_89    | 8.9                | ✅        | ✅        | ✅       |
| sm_90    | 9.0                | ✅        | ✅        | ✅       |
| sm_100   | 10.0               | ❌        | ✅        | ✅       |
| sm_120   | 12.0               | ❌        | ❌        | ✅       |


| Architecture | Compute Capability | CUDA Version | Status |
|--------------|-------------------|--------------|---------|
| Turing       | 7.5 (sm_75)      | 11.0+        | ✅      |
| Ampere       | 8.0 (sm_80)      | 11.0+        | ✅      |
| Ampere       | 8.6 (sm_86)      | 11.0+        | ✅      |
| Ada Lovelace | 8.9 (sm_89)      | 11.0+        | ✅      |
| Hopper       | 9.0 (sm_90)      | 11.0+        | ✅      |
| Hopper H200  | 10.0 (sm_100)    | 12.2+        | ✅      |
| Blackwell    | 12.0 (sm_120)    | 13.0+        | ✅      |


## Testing Considerations

### Known Limitations
1. Temperature monitoring requires nvidia-smi.exe in PATH
2. Thread termination fallback uses TerminateThread if graceful wait fails (destructors won't run on fallback termination)
3. No Jetson support (Windows-specific)

### Compatibility
- Works with all CUDA-capable NVIDIA GPUs
- Requires CUDA 11.0+ runtime
- Tested on Windows 10/11

## Future Improvements

Potential enhancements:
1. Support for more compute capabilities
2. GUI version
3. Better error messages
4. Logging to file option

## Version History

### 2025-11-08 - Initial Windows Port
- Ported from Linux to Windows
- Added sm_100 (Hopper) support
- Added sm_120 (Blackwell) support
- Created CMake build system
- Created build scripts
- Maintained feature parity with Linux version
- Fixed memory management in thread error paths
- Improved CMakeLists.txt for better CUDA support

### 2025-11-08 - Build Fixes
- Fixed Windows header conflicts: Renamed `SIZE` macro to `MATRIX_SIZE` to avoid conflicts with Windows headers
- Added Windows compatibility: Defined `ssize_t` type for Windows
- Added missing error codes: Defined `ENOMEDIUM` and `EMEDIUMTYPE` for Windows
- Fixed format strings: Changed `%ld` to `%zu` for `size_t` types, `%lld` for `ssize_t`
- Added `NOMINMAX` and `_CRT_SECURE_NO_WARNINGS` to suppress Windows-specific warnings
- Reordered includes: Windows headers now come before other includes to prevent macro conflicts
- Fixed CUDA library linking: Updated CMakeLists.txt to use `find_package(CUDAToolkit)` instead of deprecated `find_package(CUDA)`
- Fixed type conversion warnings: Added explicit casts for `size_t` to `int` and `double` to `float`
- Removed unused variable: Removed `changeCount` variable

### 2025-11-12 - Performance Display Fixes
- Fixed cursor flashing issue: Changed display update logic to only update when new data arrives (matching Linux behavior)
- Reduced CPU usage: Increased sleep interval from 100ms to 200ms in main polling loop
- Fixed performance spike: Added first update tracking to prevent initial Gflops calculation from showing unrealistic values
- Improved display stability: Removed redundant display checks that caused unnecessary screen updates
- Enhanced timing accuracy: Better handling of time delta calculations for consistent Gflops reporting

### 2025-11-12 - Critical Performance Bug Fix
- **CRITICAL**: Fixed incorrect `nonWorkIters` reset logic that caused data to be sent every 2 batches instead of every batch
- **Root cause analysis**: Linux version initializes `nonWorkIters = maxEvents` (2) and decrements each iteration, but never resets it. After first 2 warm-up iterations, it reports every batch.
- **Windows bug**: Was resetting `nonWorkIters = maxEvents` after each report, causing reporting only every 2 batches
- **Impact of fix**:
  - Performance should now match Linux version
  - Display updates frequency now matches Linux behavior
  - Eliminates cursor flashing from excessive updates
- **Additional fixes**:
  - Reduced sleep to 50ms for better responsiveness (similar to Linux select() blocking)
  - Removed unused `firstReport` variable
  - Improved temperature update responsiveness

### 2025-11-12 - Linux Version Robustness Improvements
- **Error handling**: Added robust error handling to Linux write() calls in progress reporting
- **Input validation**: Added error checking to Linux read() calls in temperature parsing
- **Resource cleanup**: Added proper pipe closing (close(writeFd)) in Linux version
- **Security improvements**: Added input validation before sscanf operations
- **Cross-platform parity**: Linux version now has comparable error handling to Windows version

### 2025-11-12 - Build System Enhancement
- **Linux-style syntax**: Added support for `COMPUTE=120` syntax in Windows build.bat
- **Backward compatibility**: Maintained support for existing `-c 120` and `--compute 120` options
- **Documentation**: Updated help to show all supported compute capabilities (75, 80, 86, 89, 90, 100, 120)
- **GPU mapping**: Added GPU architecture examples for each compute capability
- **Cross-platform consistency**: Both Linux and Windows now support the same build syntax

### 2026-05-29 - Upstream Integration & Graceful Shutdown Optimization
- **Upstream Merge**: Merged the latest commits from wilicc/gpu-burn including the memory leak fix for `d_faultyElemData` and the switch from `compare.ptx` to `compare.fatbin`.
- **Memory Leak Fix**: Ported the `cuMemFree(d_faultyElemData)` fix to `~GPU_Test()` in `win/gpu_burn-drv.cpp`.
- **Fatbin Switch**: Updated CMake build system and driver program to compile and load `compare.fatbin` instead of `compare.ptx`, aligning with upstream and allowing multi-architecture targeting.
- **Deadlock Resolution**: Removed `FlushFileBuffers` call from GPU threads which previously deadlocked them against the main thread during shutdown.
- **Graceful Thread Shutdown**: Replaced the unconditional 30-second sleep on exit with a dynamic wait using `WaitForMultipleObjects` (with sequential `WaitForSingleObject` fallback for >64 GPUs).
- **Optimization**: Marked `g_running` as `volatile` to prevent compiler register-caching optimizations, allowing child threads to terminate immediately on shutdown request.

## References

- Original GPU Burn: https://github.com/wilicc/gpu-burn
- CUDA Compute Capabilities: https://developer.nvidia.com/cuda-gpus
- CUDA Driver API: https://docs.nvidia.com/cuda/cuda-driver-api/

