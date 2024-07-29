#ifndef __LD_UNIAD_IM2COL_H__
#define __LD_UNIAD_IM2COL_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>

void cc_im2col_fwd(float * data_col, const float * data_im, const int N, const int C, const int H, const int W, 
                   const int * ksizes, const int * strides, const int * paddings, const int * dilations, 
                   const char * padding_mode) {

    int H_out = floor((float)(H + 2 * paddings[0] - dilations[0] * (ksizes[0] - 1) - 1) / (float)strides[0] + 1.0);
    int W_out = floor((float)(W + 2 * paddings[1] - dilations[1] * (ksizes[1] - 1) - 1) / (float)strides[1] + 1.0);
    
    int im2col_rows = N * H_out * W_out;
    int im2col_cols = C * ksizes[0] * ksizes[1];

    int ksize_h = ksizes[0];
    int ksize_w = ksizes[1];
    int stride_h = strides[0];
    int stride_w = strides[1];
    int pad_h = paddings[0];
    int pad_w = paddings[1];

    #pragma omp parallel for collapse(4)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h_out = 0; h_out < H_out; h_out++) {
                for (int w_out = 0; w_out < W_out; w_out++) {
                    for (int i = 0; i < ksize_h; i++) {
                        for (int j = 0; j < ksize_w; j++) {
                            int im_h = h_out * stride_h - pad_h + i;
                            int im_w = w_out * stride_w - pad_w + j;
                            int im_offset = n * C * H * W 
                                          + c * H * W 
                                          + im_h * W 
                                          + im_w;
                            int col_offset = n * H_out * W_out * im2col_cols 
                                           + (h_out * W_out + w_out) * im2col_cols 
                                           + c * ksize_h * ksize_w 
                                           + i * ksize_w 
                                           + j;
                            if (im_h >= 0 && im_h < H && im_w >= 0 && im_w < W) {
                                data_col[col_offset] = data_im[im_offset];
                            } 
                            else {
                                data_col[col_offset] = 0.0f;
                            }
                        }
                    }
                }
            }
        }
    }
#ifdef UNIAD_IM2COL_DEBUG
    printf("cc_im2col_fwd: in_shape(%d, %d, %d, %d), out_shape:(%d, %d)\n", N, C, H, W, im2col_rows, im2col_cols);
    printf("[");
    for (int i = 0; i < im2col_rows; i++) {
    // for (int i = 0; i < 256; i++) {
        printf("[");
        for (int j = 0; j < im2col_cols; j++) {
        // for (int j = 0; j < 256; j++) {
            int offset = i * im2col_cols + j;
            printf("%.3f, ", data_col[offset]);
        }
        printf("],\n");
    }
    printf("]\n");
#endif // UNIAD_IM2COL_DEBUG
}

void im2col_fwd(float * data_col, const float * data_im, const int N, const int C, const int H, const int W, 
                const int * ksizes, const int * strides, const int * paddings, const int * dilations, 
                const char * padding_mode) {
#if defined(UNIAD_BENCHMARK) || defined(UNIAD_IM2COL_BENCHMARK)
    clock_t start, end;
    double time_used;
#endif // UNIAD_BENCHMARK
    cc_im2col_fwd(data_col, data_im, N, C, H, W, ksizes, strides, paddings, dilations, padding_mode);
#if defined(UNIAD_BENCHMARK) || defined(UNIAD_IM2COL_BENCHMARK)
    end = clock();
    time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("[benchmark][im2col_fwd][cc_im2col_fwd]: NCHW:(%d, %d, %d, %d), in %f secs\n", N, C, H, W, time_used);
#endif // UNIAD_BENCHMARK
}

#endif // __LD_UNIAD_IM2COL_H__