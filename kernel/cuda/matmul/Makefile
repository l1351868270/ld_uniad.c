CC_FLAGS = -O3 -lm -Xptxas=--verbose -Xptxas=--warn-on-spills --std=c++20 -arch=sm_86 --expt-relaxed-constexpr -lcublas
CC_INCLUDES = -I../ld_kittens 
bench_gemm: bench_gemm.cu
	nvcc -o build/bench_gemm bench_gemm.cu ${CC_FLAGS} ${CC_INCLUDES}

clean:
	rm -f build/bench_gemm