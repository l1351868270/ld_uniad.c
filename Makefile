
.PHONY:test clean uniad
uniad: uniad.c
	gcc -o uniad uniad.c -Ofast -lm -fopenmp -DUNIAD_DEBUG -DUNIAD_BENCHMARK -DUNIAD_LOAD_WEIGHT_DEBUG -DUNIAD_CONV2D_DEBUG
	#  -DUNIAD_NHWC2NCHW_DEBUG  -DUNIAD_IMPAD_TO_MULTIPLE_DEBUG -DUNIAD_IMNORMALIZE_DEBUG  -DMULTIVIEW_PPM_TOFLOAT_DEBUG -DMULTIVIEW_PPM_RGB2BGR_DEBUG -DMULTIVIEW_PPM_READ_DEBUG -DPPM_TOFLOAT_DEBUG  -DPPM_RGB2BGR_DEBUG -DPPM_READ_DEBUG -DUNIAD_NHWC2NCHW_DEBUG  -DUNIAD_RESIZE_DEBUG  -DCONV2D_DEBUG -DRELU_DEBUG -DMATMUL_DEBUG -DRELU_MATMUL_DEBUG -DATAN_DEBUG -DMULTIPLY_DEBUG
	OMP_NUM_THREADS=16 ./uniad -ckpt "/root/ld_uniad.c/tools/ckpts/uniad_base_e2e.bin" -image_root "/root/ld_uniad.c/tools/data/ld_nuscenes/samples"

test:
	gcc -o test/cc/im2col_test test/cc/im2col_test.c -Ofast -lm -fopenmp -DUNIAD_TEST -DUNIAD_BENCHMARK
	OMP_NUM_THREADS=8 ./test/cc/im2col_test

clean:
	rm -f uniad test/cc/im2col_test
