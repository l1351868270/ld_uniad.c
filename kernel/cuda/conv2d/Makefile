CC_FLAGS = -O3 -lm -Xptxas=--verbose -Xptxas=--warn-on-spills --std=c++20 -arch=sm_86 --expt-relaxed-constexpr -diag-suppress 20054 -diag-suppress 128
CC_INCLUDES = -I../ld_kittens  -I../cutlass/include -I../cutlass/tools/util/include -I/root/miniconda3/envs/kittens/lib/python3.11/site-packages/include
CC_LFLAGS = -L/root/miniconda3/envs/kittens/lib/ -lcudart -lcublas -lcudnn
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