
CC_FLAGS =  -fopenmp -lm  -march=native -O3 -Wall -Wextra -Werror -Wno-deprecated-declarations -Wimplicit-fallthrough -fvisibility=hidden -fomit-frame-pointer
CC_INCLUDES = -I/usr/include/mkl
CC_LFLAGS = -L//usr/lib/x86_64-linux-gnu/

MKL_FLAGS = -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -ldl -lm
MKL_INCLUDES = -I/usr/include/mkl
MKL_LFLAGS = -L/usr/lib/x86_64-linux-gnu/

.PHONY:test clean uniad 
uniad: uniad.c
	gcc -o uniad uniad.c -Ofast ${CC_FLAGS} ${CC_INCLUDES} -DUNIAD_DEBUG -DUNIAD_BENCHMARK -DUNIAD_LOAD_WEIGHT_DEBUG -DUNIAD_CONV2D_DEBUG
	#  -DUNIAD_NHWC2NCHW_DEBUG  -DUNIAD_IMPAD_TO_MULTIPLE_DEBUG -DUNIAD_IMNORMALIZE_DEBUG  -DMULTIVIEW_PPM_TOFLOAT_DEBUG -DMULTIVIEW_PPM_RGB2BGR_DEBUG -DMULTIVIEW_PPM_READ_DEBUG -DPPM_TOFLOAT_DEBUG  -DPPM_RGB2BGR_DEBUG -DPPM_READ_DEBUG -DUNIAD_NHWC2NCHW_DEBUG  -DUNIAD_RESIZE_DEBUG  -DCONV2D_DEBUG -DRELU_DEBUG -DMATMUL_DEBUG -DRELU_MATMUL_DEBUG -DATAN_DEBUG -DMULTIPLY_DEBUG
	OMP_NUM_THREADS=8 ./uniad -ckpt "/root/ld_uniad.c/tools/ckpts/uniad_base_e2e.bin" -image_root "/root/ld_uniad.c/tools/data/ld_nuscenes/samples"


.PHONY:test_im2col
test_im2col:
	gcc -o test/cc/im2col_test test/cc/im2col_test.c -O3 -lm -fopenmp -DUNIAD_TEST -DUNIAD_BENCHMARK
	OMP_NUM_THREADS=16 ./test/cc/im2col_test


.PHONY:test_matmul
test_matmul:
	# gcc -o test/cc/matmul_test test/cc/matmul_test.c /usr/lib/x86_64-linux-gnu/atlas/libblas.so -O3 -march=native -lm -lpthread  -fopenmp -DUNIAD_TEST -DUNIAD_BENCHMARK
	# gcc -o test/cc/matmul_test test/cc/matmul_test.c -DUNIAD_TEST -DUNIAD_BENCHMARK -Wl,--as-needed -march=native -O3  -I/usr/include/mkl -L/root/miniconda3/envs/cpu/lib -fopenmp -lmkl_rt -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -liomp5 -lpthread -ldl -lm
	gcc -o test/cc/matmul_test test/cc/matmul_test.c -DUNIAD_TEST -DUNIAD_BENCHMARK -I/usr/include/mkl/ -L/usr/lib/x86_64-linux-gnu/ -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -ldl -lm
	./test/cc/matmul_test

test: test_im2col test_matmul

clean:
	rm -f uniad test/cc/im2col_test test/cc/matmul_test
