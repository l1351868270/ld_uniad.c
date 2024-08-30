CC_FLAGS = -O0 -fopenmp -lm --std=c++20
CONDA_PACKAGES_INCLUDE = $(shell python -c "import sys;import os;print(f'{os.path.dirname(sys.executable)}/../lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/include')")
$(info conda site-packages include is: $(CONDA_PACKAGES_INCLUDE))
CONDA_ENV_INCLUDE = $(shell python -c "import sys;import os;print(f'{os.path.dirname(sys.executable)}/../include')")
$(info conda env include is: $(CONDA_ENV_INCLUDE))
CC_INCLUDES = -I../ld_kittens  -I../cutlass/include -I../cutlass/tools/util/include -I$(CONDA_PACKAGES_INCLUDE) 
CC_LFLAGS = -L$(CONDA_ENV_INCLUDE)

.PHONY: implicit_gemm_test
cudnn_conv2d_test: implicit_gemm_test.cpp
	gcc -o build/implicit_gemm_test implicit_gemm_test.cpp ${CC_FLAGS} ${CC_INCLUDES} ${CC_LFLAGS}

clean:
	rm -f build/implicit_gemm_test

