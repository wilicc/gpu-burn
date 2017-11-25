// Actually, there are no rounding errors due to results being accumulated in an arbitrary order..
// Therefore EPSILON = 0.0f is OK
#define EPSILON 0.001f
#define EPSILOND 0.0000001

extern "C" __global__ void compare(float *C, int *faultyElems, size_t iters) {
	size_t iterStep = blockDim.x*blockDim.y*gridDim.x*gridDim.y;
	size_t myIndex = (blockIdx.y*blockDim.y + threadIdx.y)* // Y
		gridDim.x*blockDim.x + // W
		blockIdx.x*blockDim.x + threadIdx.x; // X

	int myFaulty = 0;
	for (size_t i = 1; i < iters; ++i)
		if (fabsf(C[myIndex] - C[myIndex + i*iterStep]) > EPSILON)
			myFaulty++;

	atomicAdd(faultyElems, myFaulty);
}

extern "C" __global__ void compareD(double *C, int *faultyElems, size_t iters) {
	size_t iterStep = blockDim.x*blockDim.y*gridDim.x*gridDim.y;
	size_t myIndex = (blockIdx.y*blockDim.y + threadIdx.y)* // Y
		gridDim.x*blockDim.x + // W
		blockIdx.x*blockDim.x + threadIdx.x; // X

	int myFaulty = 0;
	for (size_t i = 1; i < iters; ++i)
		if (fabs(C[myIndex] - C[myIndex + i*iterStep]) > EPSILOND)
			myFaulty++;

	atomicAdd(faultyElems, myFaulty);
}
