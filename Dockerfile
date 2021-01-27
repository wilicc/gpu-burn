FROM nvidia/cuda:11.1.1-devel AS builder

WORKDIR /build

COPY . /build/

RUN make

FROM nvidia/cuda:11.1.1-runtime

COPY --from=builder /build/gpu_burn /app/
COPY --from=builder /build/compare.ptx /app/

WORKDIR /app

CMD ["./gpu_burn", "60"]
