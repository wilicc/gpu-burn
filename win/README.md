# GPU Burn - Windows Version

Multi-GPU CUDA stress test for Windows

This is a Windows port of the original Linux GPU Burn tool.

## Features

- Multi-GPU support
- CUDA compute capability support: 7.5, 8.0, 8.6, 8.9, 9.0, 10.0, 12.0
- Memory stress testing
- Temperature monitoring via nvidia-smi
- Error detection and reporting
- Robust error handling and resource management
- Cross-platform build syntax compatibility

## Requirements

- Windows 10/11
- CUDA Toolkit 11.0 or later (13.0 recommended for compute 12.0 support)
- CMake 3.15 or later
- Visual Studio 2019 or later (with C++ build tools)
- NVIDIA GPU with CUDA support
- nvidia-smi (usually included with NVIDIA drivers)

## Compute Capabilities

| Architecture | Compute Capability | GPU Examples | Build Command |
|--------------|-------------------|--------------|---------------|
| Turing       | 7.5 (sm_75)       | RTX 20-series, Tesla T4 | `build.bat` |
| Ampere       | 8.0 (sm_80)       | A100, RTX 30-series | `build.bat -c 80` |
| Ampere       | 8.6 (sm_86)       | RTX 3080/3090, A30 | `build.bat -c 86` |
| Ada Lovelace | 8.9 (sm_89)       | RTX 40-series, L40 | `build.bat -c 89` |
| Hopper       | 9.0 (sm_90)       | H100 | `build.bat -c 90` |
| Hopper H200  | 10.0 (sm_100)     | H200 | `build.bat -c 100` |
| Blackwell    | 12.0 (sm_120)     | B100, B200 | `build.bat -c 120` |

Make sure to build with the appropriate compute capability for your GPU. Use `gpu_burn.exe -l` to list your GPUs and their architectures.

## Building

### Quick Build

Build with default compute capability (7.5):

```batch
build.bat
```

### Build for Specific Compute Capability

Build for specific GPU architectures:
```batch
# Turing (default)
build.bat

# Ampere
build.bat -c 80
build.bat -c 86

# Ada Lovelace
build.bat -c 89

# Hopper
build.bat -c 90
build.bat -c 100

# Blackwell
build.bat -c 120
```

### Linux-style Build Syntax

New Linux-style syntax also supported:
```batch
# Same as -c 120
build.bat COMPUTE=120

# With debug mode
build.bat COMPUTE=86 -d
```

### Manual Build with CMake

```batch
mkdir build
cd build
cmake .. -DCOMPUTE=120
cmake --build . --config Release
```

## Usage

```batch
gpu_burn.exe [OPTIONS] [TIME]
```

### Options

- `-m X` - Use X MB of memory
- `-m N%` - Use N% of the available GPU memory (default: 90%)
- `-d` - Use doubles (double precision)
- `-tc` - Try to use Tensor cores (if available)
- `-l` - List all GPUs in the system
- `-i N` - Execute only on GPU N
- `-c FILE` - Use FILE as compare kernel (default: compare.fatbin)
- `-stts T` - Set timeout threshold to T seconds (default: 30)
- `-h` - Show help message

### Examples

```batch
# Burn all GPUs for 1 hour with doubles
gpu_burn.exe -d 3600

# Burn using 50% of available GPU memory
gpu_burn.exe -m 50% 3600

# List all GPUs
gpu_burn.exe -l

# Burn only GPU 2
gpu_burn.exe -i 2 3600

# Burn with Tensor cores
gpu_burn.exe -tc 3600
```

## Differences from Linux Version

- Uses Windows threads instead of fork/exec
- Uses Windows named pipes instead of Unix pipes
- Uses Windows API for process management
- Uses Windows time functions
- Temperature monitoring still uses nvidia-smi (must be in PATH)
- Polling-based event loop (50ms sleep) vs Linux select() event-driven
- Enhanced error handling and resource cleanup
- Cross-platform build syntax support (both `-c 120` and `COMPUTE=120`)

## Troubleshooting

### nvidia-smi not found
Make sure nvidia-smi.exe is in your PATH. It's usually located in:
`C:\Program Files\NVIDIA Corporation\NVSMI\`

### CUDA not found
Make sure CUDA Toolkit is installed and nvcc is in your PATH.

### Build errors
- Ensure Visual Studio C++ build tools are installed
- Check that CMake can find CUDA: `cmake .. -DCUDA_TOOLKIT_ROOT_DIR="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"`

## License

Same as original GPU Burn project (BSD-style license)

