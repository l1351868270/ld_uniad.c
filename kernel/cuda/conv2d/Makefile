CC_FLAGS = -O3 -lm -Xptxas=--verbose -Xptxas=--warn-on-spills --std=c++20 -arch=sm_86 --expt-relaxed-constexpr -diag-suppress 20054 -diag-suppress 128
CONDA_PACKAGES_INCLUDE = $(shell python -c "import sys;import os;print(f'{os.path.dirname(sys.executable)}/../lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/include')")
$(info conda site-packages include is: $(CONDA_PACKAGES_INCLUDE))
CONDA_ENV_INCLUDE = $(shell python -c "import sys;import os;print(f'{os.path.dirname(sys.executable)}/../include')")
$(info conda env include is: $(CONDA_ENV_INCLUDE))
CC_INCLUDES = -I../ld_kittens  -I../cutlass/include -I../cutlass/tools/util/include -I$(CONDA_PACKAGES_INCLUDE) 
CC_LFLAGS = -L$(CONDA_ENV_INCLUDE) -lcudart -lcublas -lcudnn

.PHONY: cudnn_conv2d_test
cudnn_conv2d_test: cudnn_conv2d_test.cu
	nvcc -o build/cudnn_conv2d_test cudnn_conv2d_test.cu  ${CC_FLAGS} ${CC_INCLUDES} ${CC_LFLAGS}

.PHONY: cutlass_implicit_gemm_test
cutlass_implicit_gemm_test: cutlass_implicit_gemm_test.cu
	nvcc -o build/cutlass_implicit_gemm_test cutlass_implicit_gemm_test.cu  ${CC_FLAGS} ${CC_INCLUDES} ${CC_LFLAGS}

.PHONY: cudnn_frontend_test
cudnn_frontend_test: cudnn_frontend_test.cu
	nvcc -o build/cudnn_frontend_test cudnn_frontend_test.cu  ${CC_FLAGS} ${CC_INCLUDES} ${CC_LFLAGS}

clean:
	rm -f build/cudnn_conv2d_test build/cutlass_implicit_gemm_test build/cudnn_frontend_test