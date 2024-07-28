#ifndef __LD_UNIAD_MAXPOOL2D_H__
#define __LD_UNIAD_MAXPOOL2D_H__

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

// https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
// format NCHW
void cc_maxpool2d_fwd(float * output, int * output_shape, float * input, int * input_shape,
                      int * kernel_sizes, int * strides, int * paddings, const char * padding_mode) {
    int N = input_shape[0];
    int C = input_shape[1];
    int H_in = input_shape[2];
    int W_in = input_shape[3];
    int H_out = output_shape[2];
    int W_out = output_shape[3];

    int kernel_size_h = kernel_sizes[0];
    int kernel_size_w = kernel_sizes[1];

    int stride_h = strides[0];
    int stride_w = strides[1];

    int padding_h = paddings[0];
    int padding_w = paddings[1];

    #pragma omp parallel for collapse(4)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h_out = 0; h_out < H_out; h_out++) {
                for (int w_out = 0; w_out < W_out; w_out++) {
                    float out_val = 0.0f;
                    int out_offset = n * C * H_out * W_out
                                   + c * H_out * W_out
                                   + h_out * W_out
                                   + w_out;
                    float max_val = -INFINITY;
                    for (int i = 0; i < kernel_size_h; i++) {
                        for (int j = 0; j < kernel_size_w; j++) {
                            int in_offset = n * C * H_in * W_in
                                          + c * H_in * W_in
                                          + (stride_h * h_out + i - padding_h) * W_in
                                          + (stride_w * w_out + j - padding_w);
                            float input_val = 0.0f;
                            if (stride_h * h_out + i >= padding_h && stride_h * h_out + i < H_in + 2 * padding_h && 
                                stride_w * w_out + j >= padding_w && stride_w * w_out + j < W_in + 2 * padding_w) {
                                input_val = input[in_offset];
                            }
                            if (input_val > max_val) {
                                max_val = input_val;
                            }
                        }
                    }
                    output[out_offset] = max_val;
                }
            }
        }
    }
#ifdef UNIAD_MAXPOOL2D_DEBUG
    printf("maxpool2d_fwd: (%d, %d, %d, %d)\n", N, C, H_out, W_out);
    printf("[");
    for (int n = 0; n < N; n++) {
        printf("[");
        for (int c = 0; c < C; c++) {
            printf("[");
            // for (int i = 0; i < out_height; i++) {
            for (int h = 0; h < 16; h++) {
                // printf("(%d, %d, %d): [", n, c, h);
                printf("[");
                // for (int j = 0; j < out_width; j++) {
                for (int w = 0; w < 16; w++) {
                    int offset = n * C * H_out * W_out
                                   + c * H_out * W_out
                                   + h * W_out
                                   + w;
                    printf("%.3f, ", output[offset]);
                }
                printf("],\n");
            }
            printf("],\n");
        }
        printf("],\n");
    }
    printf("]\n");
#endif
}

void maxpool2d_fwd(float * output, int * output_shape, float * input, int * input_shape,
                int * kernel_sizes, int * strides, int * paddings, const char * padding_mode) {
#if defined(UNIAD_BENCHMARK) || defined(UNIAD_MAXPOOL2D_BENCHMARK)
    double tdata = omp_get_wtime();
#endif // UNIAD_BENCHMARK
    cc_maxpool2d_fwd(output, output_shape, input, input_shape, kernel_sizes, strides, paddings, padding_mode);
#if defined(UNIAD_BENCHMARK) || defined(UNIAD_MAXPOOL2D_BENCHMARK)
    tdata = omp_get_wtime() - tdata;
    printf("[benchmark][maxpool2d_fwd][cc_maxpool2d_fwd]: in_shape:(%d, %d, %d, %d), out_shape:(%d, %d, %d, %d), "
           "strides:(%d, %d), panddings:(%d, %d) in %f secs\n", 
           input_shape[0], input_shape[1], input_shape[2], input_shape[3], 
           output_shape[0], output_shape[1], output_shape[2], output_shape[3], 
           strides[0], strides[1], paddings[0], paddings[1], tdata);
#endif // UNIAD_BENCHMARK
}

#endif // UNIAD_MAXPOOL2D_DEBUG