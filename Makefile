TARGET := NVIDIA
# TARGET := AMD

ifneq ("$(wildcard /opt/cuda/bin/nvcc)", "")
CUDAPATH ?= /opt/cuda
else ifneq ("$(wildcard /usr/local/cuda/bin/nvcc)", "")
CUDAPATH ?= /usr/local/cuda
endif

IS_JETSON   ?= $(shell if grep -Fwq "Jetson" /proc/device-tree/model 2>/dev/null; then echo true; else echo false; fi)
NVCC        :=  ${CUDAPATH}/bin/nvcc
CCPATH      ?=
ROCM_PATH   := /opt/rocm

override CFLAGS   ?=
override CFLAGS   += -O3
override CFLAGS   += -Wno-unused-result
override CFLAGS   += -std=c++11

ifeq ($(TARGET),NVIDIA)
override CFLAGS += -D__TARGET_NVIDIA
endif
ifeq ($(TARGET),AMD)
override CFLAGS += -D__TARGET_AMD
endif

ifeq ($(TARGET),NVIDIA)
override CFLAGS   += -I${CUDAPATH}/include
override CFLAGS   += -DIS_JETSON=${IS_JETSON}
endif

override LDFLAGS  ?=

ifeq ($(TARGET),NVIDIA)
override LDFLAGS  += -lcuda
override LDFLAGS  += -L${CUDAPATH}/lib64
override LDFLAGS  += -L${CUDAPATH}/lib64/stubs
override LDFLAGS  += -L${CUDAPATH}/lib
override LDFLAGS  += -L${CUDAPATH}/lib/stubs
override LDFLAGS  += -Wl,-rpath=${CUDAPATH}/lib64
override LDFLAGS  += -Wl,-rpath=${CUDAPATH}/lib
override LDFLAGS  += -lcublas
override LDFLAGS  += -lcudart
endif

ifeq ($(TARGET),AMD)
override LDFLAGS  += -Wl,-rpath=$(ROCM_PATH)/lib -lhipblas
override CFLAGS += -I$(ROCM_PATH)/include
override CFLAGS += -I$(ROCM_PATH)/include/hipblas
override CFLAGS += -D__HIP_PLATFORM_AMD__
endif

COMPUTE      ?= 75
CUDA_VERSION ?= 11.8.0
IMAGE_DISTRO ?= ubi8

override NVCCFLAGS ?=
override NVCCFLAGS += -I${CUDAPATH}/include
override NVCCFLAGS += -arch=compute_$(subst .,,${COMPUTE})

IMAGE_NAME ?= gpu-burn

.PHONY: clean

ifeq ($(TARGET),AMD)

all: gpu_burn_hip compare.o

gpu_burn_hip: gpu_burn-drv_hip.o
	hipcc $(CFLAGS) $^ $(LDFLAGS) -o $@

%.hip: %.cu
	hipify-perl $< -o $@

gpu_burn-drv_hip.cpp: gpu_burn-drv.cpp
	hipify-perl $< -o $@

%.o: %.hip
	hipcc --genco $(CFLAGS) $< -o $@

%.o: %.cpp
	hipcc ${CFLAGS} -c $<

endif # TARGET AMD

ifeq ($(TARGET),NVIDIA)

gpu_burn: gpu_burn-drv.o compare.ptx
	g++ -o $@ $< -O3 ${LDFLAGS}

%.o: %.cpp
	g++ ${CFLAGS} -c $<

%.ptx: %.cu
	PATH="${PATH}:${CCPATH}:." ${NVCC} ${NVCCFLAGS} -ptx $< -o $@

endif # TARGET NVIDIA

clean:
	$(RM) *.hip *.ptx *.o gpu_burn gpu_burn_hip

image:
	docker build --build-arg CUDA_VERSION=${CUDA_VERSION} --build-arg IMAGE_DISTRO=${IMAGE_DISTRO} -t ${IMAGE_NAME} .
