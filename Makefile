CUDAPATH=/usr/local/cuda

# Have this point to an old enough gcc (for nvcc)
GCCPATH=/usr

NVCC=${CUDAPATH}/bin/nvcc
CCPATH=${GCCPATH}/bin

COMPUTE ?= 50

.PHONY: clean

gpu_burn: gpu_burn-drv.o compare.ptx
	g++ -o gpu_burn gpu_burn-drv.o -O3 -lcuda -L${CUDAPATH}/lib64 -L${CUDAPATH}/lib -Wl,-rpath=${CUDAPATH}/lib64 -Wl,-rpath=${CUDAPATH}/lib -lcublas -lcudart -o gpu_burn

gpu_burn-drv.o: gpu_burn-drv.cpp
	g++ -O3 -Wno-unused-result -I${CUDAPATH}/include -c $<

compare.ptx: compare.cu
	PATH=${PATH}:.:${CCPATH}:${PATH} ${NVCC} -I${CUDAPATH}/include -arch=compute_$(subst .,,${COMPUTE}) -ptx $< -o $@

clean:
	$(RM) *.ptx *.o gpu_burn
