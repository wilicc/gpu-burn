ARG CUDA_VERSION=11.8.0
ARG IMAGE_DISTRO=ubi8
ARG COMPUTE=75

FROM nvidia/cuda:${CUDA_VERSION}-devel-${IMAGE_DISTRO} AS builder

ARG COMPUTE=75

WORKDIR /build

COPY compare.cu gpu_burn-drv.cpp Makefile /build/

RUN make COMPUTE=${COMPUTE}

FROM nvidia/cuda:${CUDA_VERSION}-runtime-${IMAGE_DISTRO}

COPY --from=builder /build/gpu_burn /app/
COPY --from=builder /build/compare.fatbin /app/

WORKDIR /app

ENTRYPOINT ["./gpu_burn"]
CMD ["60"]
