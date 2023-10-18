# gpu-burn
Multi-GPU CUDA stress test
http://wili.cc/blog/gpu-burn.html

# Easy docker build and run

```
git clone https://github.com/wilicc/gpu-burn
cd gpu-burn
docker build -t gpu_burn .
docker run --rm --gpus all gpu_burn
```

# Building
To build GPU Burn:

`make CUDA_PATH=<path/to/cuda-<version>`

To build GPU Burn with EasyBuild module env loaded:

`make`

To remove artifacts built by GPU Burn:

`make clean`

GPU Burn builds with a default Compute Capability of 5.0.
To override this with a different value:

`make CUDA_PATH=<path/to/cuda-<version> COMPUTE=<compute capability value>`

CFLAGS can be added when invoking make to add to the default
list of compiler flags:

`make CUDA_PATH=<path/to/cuda-<version> CFLAGS=-Wall`

LDFLAGS can be added when invoking make to add to the default
list of linker flags:

`make CUDA_PATH=<path/to/cuda-<version> LDFLAGS=-lmylib`

NVCCFLAGS can be added when invoking make to add to the default
list of nvcc flags:

`make CUDA_PATH=<path/to/cuda-<version> NVCCFLAGS=-ccbin <path to host compiler>`

CCPATH can be specified to point to a specific gcc (default is
/usr/bin):

`make CUDA_PATH=<path/to/cuda-<version> CCPATH=/usr/local/bin`

CUDA_VERSION and IMAGE_DISTRO can be used to override the base
images used when building the Docker `image` target, while IMAGE_NAME
can be set to change the resulting image tag:

`make IMAGE_NAME=myregistry.private.com/gpu-burn CUDA_VERSION=12.0.1 IMAGE_DISTRO=ubuntu22.04 image`

# Usage

    GPU Burn
    Usage: gpu_burn [OPTIONS] [TIME]
    
    -m X   Use X MB of memory
    -m N%  Use N% of the available GPU memory
    -d     Use doubles
    -tc    Try to use Tensor cores (if available)
    -l     List all GPUs in the system
    -i N   Execute only on GPU N
    -h     Show this help message
    
    Example:
    gpu_burn -d 3600
