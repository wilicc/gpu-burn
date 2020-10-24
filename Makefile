CUDAPATH=/usr/local/cuda

NVCC=${CUDAPATH}/bin/nvcc
CCPATH ?=

CFLAGS  ?=
CFLAGS  += -O3
CFLAGS  += -Wno-unused-result
CFLAGS  += -I${CUDAPATH}/include

LDFLAGS ?=
LDFLAGS += -lcuda
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

%.o: %.cpp
	g++ ${CFLAGS} -c $<

%.ptx: %.cu
	PATH=${PATH}:${CCPATH}:. ${NVCC} -I${CUDAPATH}/include -arch=compute_$(subst .,,${COMPUTE}) -ptx $< -o $@

clean:
	$(RM) *.ptx *.o gpu_burn
