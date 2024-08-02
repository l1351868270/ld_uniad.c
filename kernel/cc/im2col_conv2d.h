#ifndef __LD_UNIAD_IM2COL_CONV2D_H__
#define __LD_UNIAD_IM2COL_CONV2D_H__

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>

// https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
// format NCHW
void cc_im2col_conv2d_fwd(float * output, int * output_shape, float * input, int * input_shape,
                float * weight, int * weight_shape, float * bias, 
                int * strides, int * paddings, const char * padding_mode) {
    if (strcmp(padding_mode, "zeros")) {
        fprintf(stderr, "[cc_im2col_conv2d_fwd]: padding_mode only suppoer zeros\n");
    }
    int N = input_shape[0];
    int C_in = input_shape[1];
    int H_in = input_shape[2];
    int W_in = input_shape[3];
    int C_out = output_shape[1];
    int H_out = output_shape[2];
    int W_out = output_shape[3];

    int kernel_size_h = weight_shape[2];
    int kernel_size_w = weight_shape[3];

    int stride_h = strides[0];
    int stride_w = strides[1];

    // int weight_h = weight_shape[2];
    // int weight_w = weight_shape[3];

    int padding_h = paddings[0];
    int padding_w = paddings[1];

    #pragma omp parallel for collapse(4)
    for (int n = 0; n < N; n++) {
        for (int c_out = 0; c_out < C_out; c_out++) {
            for (int h_out = 0; h_out < H_out; h_out++) {
                for (int w_out = 0; w_out < W_out; w_out++) {
                    float out_val = 0.0f;
                    int bias_offset = c_out;
                    int out_offset = n * C_out * H_out * W_out
                                   + c_out * H_out * W_out
                                   + h_out * W_out
                                   + w_out;
                    for (int c_in = 0; c_in < C_in; c_in++) {
                        for (int i = 0; i < kernel_size_h; i++) {
                            for (int j = 0; j < kernel_size_w; j++) {
                                int h_in = h_out * stride_h + i - padding_h;
                                int w_in = w_out * stride_w + j - padding_w;
                                int in_offset = n * C_in * H_in * W_in
                                              + c_in * H_in * W_in
                                              + h_in * W_in
                                              + w_in;
                                int weight_offset = c_out * C_in * kernel_size_h * kernel_size_w
                                                  + c_in * kernel_size_h * kernel_size_w
                                                  + i * kernel_size_w
                                                  + j;
                                float input_val = 0.0f;
                                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                                    input_val = input[in_offset];
                                }
                                out_val += input_val * weight[weight_offset];
                            }
                        }
                    }
                    output[out_offset] = out_val;
                    if (bias != NULL) {
                        output[out_offset] += bias[bias_offset];
                    }
                }
            }
        }
    }
#ifdef UNIAD_IM2COL_CONV2D_DEBUG
    printf("[im2col_conv2d_fwd]: (%d, %d, %d, %d)\n", N, C_out, H_out, W_out);
    printf("[");
    for (int n = 0; n < N; n++) {
        printf("[");
        for (int c = 0; c < C_out; c++) {
            printf("[");
            // for (int i = 0; i < out_height; i++) {
            for (int h = 0; h < 16; h++) {
                // printf("(%d, %d, %d): [", n, c, h);
                printf("[");
                // for (int j = 0; j < out_width; j++) {
                for (int w = 0; w < 16; w++) {
                    int offset = n * C_out * H_out * W_out
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

    printf("[im2col_conv2d_fwd] weight: \n");
    printf("[");
    for (int n = 0; n < C_out; n++) {
        printf("[");
        for (int c = 0; c < C_in; c++) {
            printf("[");
            // for (int i = 0; i < out_height; i++) {
            for (int h = 0; h < kernel_size_h; h++) {
                printf("(%d, %d, %d): [", n, c, h);
                // for (int j = 0; j < out_width; j++) {
                for (int w = 0; w < kernel_size_w; w++) {
                    int offset = n * C_in * kernel_size_h * kernel_size_w
                                   + c * kernel_size_h * kernel_size_w
                                   + h * kernel_size_w
                                   + w;
                    printf("%.3f, ", weight[offset]);
                }
                printf("],\n");
            }
            printf("],\n");
        }
        printf("],\n");
    }
    printf("]\n");


    printf("[im2col_conv2d_fwd] input: \n");
    printf("[");
    for (int n = 0; n < N; n++) {
        printf("[");
        for (int c = 0; c < C_in; c++) {
            printf("[");
            // for (int i = 0; i < out_height; i++) {
            for (int h = 0; h < 16; h++) {
                printf("(%d, %d, %d): [", n, c, h);
                // for (int j = 0; j < out_width; j++) {
                for (int w = 0; w < 16; w++) {
                    int offset = n * C_in * H_in * W_in
                                   + c * H_in * W_in
                                   + h * W_in
                                   + w;
                    printf("%.3f, ", input[offset]);
                }
                printf("],\n");
            }
            printf("],\n");
        }
        printf("],\n");
    }
    printf("]\n");
#endif // UNIAD_IM2COL_CONV2D_DEBUG
}

void im2col_conv2d_fwd(float * output, int * output_shape, float * input, int * input_shape,
                float * filters, int * filters_shape, float * bias, 
                int * strides, int * paddings, const char * padding_mode) {
#if defined(UNIAD_BENCHMARK) || defined(UNIAD_IM2COL_CONV2D_BENCHMARK)
    double tdata = omp_get_wtime();
#endif // UNIAD_BENCHMARK
    cc_im2col_conv2d_fwd(output, output_shape, input, input_shape, filters, filters_shape, bias, strides, paddings, padding_mode);
#if defined(UNIAD_BENCHMARK) || defined(UNIAD_IM2COL_CONV2D_BENCHMARK)
    tdata = omp_get_wtime() - tdata;
    printf("[benchmark][im2col_conv2d_fwd][cc_im2col_conv2d_fwd]: in_shape:(%d, %d, %d, %d), weight_shape:(%d, %d, %d, %d), out_shape:(%d, %d, %d, %d), "
           "strides:(%d, %d), panddings:(%d, %d) in %f secs\n", 
           input_shape[0], input_shape[1], input_shape[2], input_shape[3],
           filters_shape[0], filters_shape[1], filters_shape[2], filters_shape[3],
           output_shape[0], output_shape[1], output_shape[2], output_shape[3], 
           strides[0], strides[1], paddings[0], paddings[1], tdata);
#endif // UNIAD_BENCHMARK
}

#endif // UNIAD_IM2COL_CONV2D_DEBUG

