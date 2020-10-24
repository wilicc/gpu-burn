CUDAPATH=/usr/local/cuda

# Have this point to an old enough gcc (for nvcc)
GCCPATH=/usr

NVCC=${CUDAPATH}/bin/nvcc
CCPATH=${GCCPATH}/bin

LDFLAGS := -lcuda
LDFLAGS += -L${CUDAPATH}/lib64
LDFLAGS += -L${CUDAPATH}/lib
LDFLAGS += -Wl,-rpath=${CUDAPATH}/lib64
LDFLAGS += -Wl,-rpath=${CUDAPATH}/lib
LDFLAGS += -lcublas
LDFLAGS += -lcudart

COMPUTE ?= 50

.PHONY: clean

gpu_burn: gpu_burn-drv.o compare.ptx
	g++ -o $@ $< -O3 ${LDFLAGS}

gpu_burn-drv.o: gpu_burn-drv.cpp
	g++ -O3 -Wno-unused-result -I${CUDAPATH}/include -c $<

compare.ptx: compare.cu
	PATH=${PATH}:.:${CCPATH}:${PATH} ${NVCC} -I${CUDAPATH}/include -arch=compute_$(subst .,,${COMPUTE}) -ptx $< -o $@

clean:
	$(RM) *.ptx *.o gpu_burn
