# gpu-burn
Multi-GPU CUDA stress test
http://wili.cc/blog/gpu-burn.html

# Building
To build GPU Burn:

`make`

To remove artifacts built by GPU Burn:

`make clean`

GPU Burn builds with a default Compute Capability of 5.0.
To override this with a different value:

`make COMPUTE=<compute capability value>`

CFLAGS can be added when invoking make to add to the default
list of compiler flags:

`make CFLAGS=-Wall`

LDFLAGS can be added when invoking make to add to the default
list of linker flags:

`make LDFLAGS=-lmylib`

NVCCFLAGS can be added when invoking make to add to the default
list of nvcc flags:

`make NVCCFLAGS=-ccbin <path to host compiler>`

CUDAPATH can be added to point to a non standard install or
specific version of the cuda toolkit (default is 
/usr/local/cuda):

`make CUDAPATH=/usr/loca/cuda-<version>`

CCPATH can be specified to point to a specific gcc (default is
/usr/bin):

`make CCPATH=/usr/local/bin`

# Usage

    GPU Burn
    Usage: gpu_burn [OPTIONS] [TIME]
    
    -d	Use doubles
    -tc	Use Tensor cores
    -h	Show this help message
    
    Example:
    gpu-burn -d 3600
