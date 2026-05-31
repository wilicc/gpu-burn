# gpu-burn (Windows)

Windows port of [gpu-burn](https://github.com/wilicc/gpu-burn).
Same stress-test behaviour as the Linux version; differences are limited
to the host process model (Windows threads + named pipes instead of
`fork(2)` + Unix pipes) and the build system.

## Requirements

- Windows 10/11
- CUDA Toolkit 11.0+ (13.0+ for `compute_120` / Blackwell)
- CMake 3.18+
- Visual Studio 2019 or 2022 with the C++ build tools
- An NVIDIA driver with `nvidia-smi` on `PATH`

## Building

```batch
build.bat                  REM defaults to compute_75 (Turing)
build.bat -c 86            REM Ampere
build.bat -c 89            REM Ada
build.bat -c 90            REM Hopper
build.bat -c 120           REM Blackwell
build.bat -c 86 -d         REM Debug build
```

The `compare.cu` kernel is shared with the Linux build and lives at the
repository root. `build.bat` is a thin wrapper around CMake; equivalent
manual invocation:

```batch
cmake -S . -B build -DCOMPUTE=120
cmake --build build --config Release
```

The resulting `gpu_burn.exe` and `compare.fatbin` are placed in
`build\Release\` (or `build\Debug\`).

## Usage

Same flags as the Linux binary. See the root [README](../README.md) and
`gpu_burn.exe -h`.

```batch
gpu_burn.exe -d 3600       REM burn all GPUs with doubles for an hour
gpu_burn.exe -m 50% 60     REM 50% of available VRAM, 1 minute
gpu_burn.exe -i 0 60       REM only GPU 0
gpu_burn.exe -l            REM list GPUs and exit
```
