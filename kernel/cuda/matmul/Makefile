CC_FLAGS = -O3 -lm -Xptxas=--verbose -Xptxas=--warn-on-spills --std=c++20 -arch=sm_86 --expt-relaxed-constexpr -diag-suppress 20054 -lcublas
CC_INCLUDES = -I../ld_kittens  -I../cutlass/include -I../cutlass/tools/util/include
bench_gemm: bench_gemm.cu
	nvcc -o build/bench_gemm bench_gemm.cu ${CC_FLAGS} ${CC_INCLUDES}

.PHONY: cutlass_gemm_v2_test
cutlass_gemm_v2_test: cutlass_gemm_v2_test.cu
	nvcc -o build/cutlass_gemm_v2_test cutlass_gemm_v2_test.cu ${CC_FLAGS} ${CC_INCLUDES}

.PHONY: test
test: bench_gemm
	./build/bench_gemm

clean:
	rm -f build/bench_gemm