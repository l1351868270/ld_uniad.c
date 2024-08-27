CONDA_PACKAGES_INCLUDE = $(shell python -c "import sys;import os;print(f'{os.path.dirname(sys.executable)}/../lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/include')")
$(info conda site-packages include is: $(CONDA_PACKAGES_INCLUDE))
CONDA_ENV_INCLUDE = $(shell python -c "import sys;import os;print(f'{os.path.dirname(sys.executable)}/../include')")
$(info conda env include is: $(CONDA_ENV_INCLUDE))

CC_FLAGS = -O3 -lm -Xptxas=--verbose -Xptxas=--warn-on-spills --std=c++20 -arch=sm_86 --expt-relaxed-constexpr -diag-suppress 20054 -diag-suppress 128 -lcudart -lcublas -lcudnn
CC_INCLUDES = -I../ld_kittens  -I../cutlass/include -I../cutlass/tools/util/include -I$(CONDA_PACKAGES_INCLUDE) 
bench_gemm: bench_gemm.cu
	nvcc -o build/bench_gemm bench_gemm.cu ${CC_FLAGS} ${CC_INCLUDES}

.PHONY: cutlass_gemm_v2_test
cutlass_gemm_v2_test: cutlass_gemm_v2_test.cu
	nvcc -o build/cutlass_gemm_v2_test cutlass_gemm_v2_test.cu ${CC_FLAGS} ${CC_INCLUDES}

.PHONY: cudnn_frontend_gemm_test
cudnn_frontend_gemm_test: cudnn_frontend_gemm_test.cu
	nvcc -o build/cudnn_frontend_gemm_test cudnn_frontend_gemm_test.cu ${CC_FLAGS} ${CC_INCLUDES}

.PHONY: test
test: bench_gemm
	./build/bench_gemm

clean:
	rm -f build/bench_gemm