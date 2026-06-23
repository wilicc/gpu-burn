# gpu-burn

Multi-GPU CUDA stress test
<http://wili.cc/blog/gpu-burn.html>

- [gpu-burn](#gpu-burn)
  - [Easy docker build and run](#easy-docker-build-and-run)
  - [Binary packages](#binary-packages)
  - [Building](#building)
  - [Usage](#usage)

## Easy docker build and run

```plain
git clone https://github.com/wilicc/gpu-burn
cd gpu-burn
docker build -t gpu-burn .
docker run --rm --gpus all gpu-burn
```
You can pass build arguments to specify the CUDA version, the compute capability, and base image distro, e.g.:

```plain
docker build --build-arg CUDA_VERSION=13.0.0 --build-arg COMPUTE=75 --build-arg IMAGE_DISTRO=ubi8 -t gpu-burn .
```
## Binary packages

<https://repology.org/project/gpu-burn/versions>

## Building

To build GPU Burn:

`make`

To remove artifacts built by GPU Burn:

`make clean`

GPU Burn builds with a default Compute Capability of 7.5 as specified on NVIDIA's [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda-gpus).
To override this with a different value:

`make COMPUTE=<compute capability value>`

`COMPUTE` selects a single virtual architecture for the default `-arch`
flag.  For a fat binary targeting multiple architectures, or to opt into
architecture-conditional (`sm_90a`) or family-specific (`compute_100f`)
features, set `COMPUTE=` to suppress the default and drive the build
entirely from `-gencode` flags via `NVCCFLAGS`:

`make COMPUTE= NVCCFLAGS='-gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_90,code=sm_90'`

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

`make CUDAPATH=/usr/local/cuda-<version>`

CCPATH can be specified to point to a specific gcc (default is
/usr/bin):

`make CCPATH=/usr/local/bin`

CUDA_VERSION and IMAGE_DISTRO can be used to override the base
images used when building the Docker `image` target, while IMAGE_NAME
can be set to change the resulting image tag:

`make IMAGE_NAME=myregistry.private.com/gpu-burn CUDA_VERSION=12.0.1 IMAGE_DISTRO=ubuntu22.04 image`

## Usage

```plain
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
```

### Large single-allocation timeout on unified-memory GPUs (GH/GB)

On Grace-Blackwell / Grace-Hopper unified-memory parts (e.g. GB200/GB300),
gpu-burn allocates the result buffer as **one** giant `cuMemAlloc`
(`useBytes`, i.e. ~90% of memory by default — hundreds of GB). On these parts
that single allocation can make the driver's hardware memory scrubber time out
(`NVRM ... mem_scrub.c`, 30s limit) and the kernel soft-lock in `clear_page`;
`cuMemAlloc` then returns out-of-memory (reported as `C alloc`) and the run
DIEs — reproducible even without any concurrent memory pressure.

Set `GPU_BURN_MAX_ALLOC_GB` to cap the single allocation so the block stays
small enough to be scrubbed within the timeout:

```plain
GPU_BURN_MAX_ALLOC_GB=24 gpu_burn -tc 3600
```

Unset or `<=0` keeps the default allocate-everything behaviour. Per-iteration
GEMM stress is unchanged (still full `SIZE^3`); only the number of result-buffer
iterations shrinks, so compute/thermal/power load is effectively the same.
