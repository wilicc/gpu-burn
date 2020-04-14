CUDAPATH=/usr/local/cuda

# Have this point to an old enough gcc (for nvcc)
GCCPATH=/usr

NVCC=${CUDAPATH}/bin/nvcc
CCPATH=${GCCPATH}/bin
CC=g++

drv:
	PATH=${PATH}:.:${CCPATH}:${PATH} ${NVCC} -ccbin ${CC} -I${CUDAPATH}/include -arch=compute_30 -ptx compare.cu -o compare.ptx
	${CC} -O3 -Wno-unused-result -I${CUDAPATH}/include -c gpu_burn-drv.cpp
	${CC} -o gpu_burn gpu_burn-drv.o -O3 -lcuda -L${CUDAPATH}/lib64 -L${CUDAPATH}/lib -Wl,-rpath=${CUDAPATH}/lib64 -Wl,-rpath=${CUDAPATH}/lib -lcublas -lcudart -o gpu_burn
