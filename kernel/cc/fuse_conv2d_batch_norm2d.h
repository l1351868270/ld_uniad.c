#ifndef __LD_UNIAD_FUSE_CONV2D_BATCH_NORM2D_H__
#define __LD_UNIAD_FUSE_CONV2D_BATCH_NORM2D_H__

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

// format NCHW
void cc_fuse_conv2d_batch_norm2d_fwd(float * output, int * output_shape, float * input, int * input_shape,
                float * weight, int * weight_shape, float * bias, 
                int * strides, int * paddings, const char * padding_mode,
                const float * gamma, const float * beta, const float * mean, const float * var) {
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

    int weight_h = weight_shape[2];
    int weight_w = weight_shape[3];

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
                                int in_offset = n * C_in * H_in * W_in
                                              + c_in * H_in * W_in
                                              + (stride_h * h_out + i - padding_h) * W_in
                                              + (stride_w * w_out + j - padding_w);
                                int weight_offset = c_out * C_in * kernel_size_h * kernel_size_w
                                                  + c_in * kernel_size_h * kernel_size_w
                                                  + i * kernel_size_w
                                                  + j;
                                float input_val = 0.0f;
                                if (stride_h * h_out + i >= padding_h && stride_h * h_out + i < H_in + 2 * padding_h && 
                                    stride_w * w_out + j >= padding_w && stride_w * w_out + j < W_in + 2 * padding_w) {
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
                    output[out_offset] = (output[out_offset] - mean[c_out]) / sqrtf(var[c_out] + 1e-5) * gamma[c_out] + beta[c_out];
                }
            }
        }
    }
#ifdef UNIAD_FUSE_CONV2D_BATCH_NORM2D_DEBUG
    printf("fuse_conv2d_batch_norm2d_fwd: (%d, %d, %d, %d)\n", N, C_out, H_out, W_out);
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
#endif // UNIAD_FUSE_CONV2D_BATCH_NORM2D_DEBUG
}

// void cc_fuse_conv2d_batch_norm2d_fwd(float * output, int * output_shape, float * input, int * input_shape,
//                 float * weight, int * weight_shape, float * bias, 
//                 int * strides, int * paddings, const char * padding_mode) {
//     int N = input_shape[0];
//     int C_in = input_shape[1];
//     int H_in = input_shape[2];
//     int W_in = input_shape[3];
//     int C_out = output_shape[1];
//     int H_out = output_shape[2];
//     int W_out = output_shape[3];
    
//     int padding = paddings[0];
//     int kernel_size = weight_shape[2];
//     int stride = strides[0];

//     #pragma omp parallel for collapse(4)
//     for (int n = 0; n < N; n++) {
//         for (int c_out = 0; c_out < C_out; c_out++) {
//             for (int h_in = 0; h_in < H_in + 2 * padding - kernel_size + 1; h_in += stride) {
//                 for (int w_in = 0; w_in < W_in + 2 * padding - kernel_size + 1; w_in += stride) {
//                     int offset_out = n * C_out * H_out * W_out
//                                    + c_out * H_out * W_out 
//                                    + h_in / stride * W_out
//                                    + w_in / stride;
//                     float value = 0.0f;
//                     for (int c_in = 0; c_in < C_in; c_in++) {
//                         for (int k_i = 0; k_i < kernel_size; k_i++) {
//                             for (int k_j = 0; k_j < kernel_size; k_j++){
//                                 int offset_kernel = c_out * C_in * kernel_size * kernel_size
//                                                   + c_in * kernel_size * kernel_size
//                                                   + k_i * kernel_size + k_j;
//                                 float input_v = 0.0f;
//                                 if (h_in + k_i >= padding && h_in + k_i < H_in + padding && w_in + k_j >= padding && w_in + k_j < W_in + padding) {
//                                     int offset_in = n * C_in * H_in * W_in
//                                                   + c_in * H_in * W_in
//                                                   + (h_in - padding) * W_in
//                                                   + (w_in - padding)
//                                                   + k_i * W_in + k_j;
//                                     input_v = input[offset_in];
                                
//                                 }
//                                 value += input_v * (*(weight + offset_kernel));
//                             }
//                         }                   
//                     }
//                     output[offset_out] = value;
//                     if (bias != NULL) {
//                         output[offset_out] += bias[c_out];
//                     }
//                     // if (offset_out < N * C_out * H_out * W_out && offset_out >= N * C_out * H_out * W_out - 640) {
//                     // if (offset_out < 640) {
//                     //     printf("fuse_conv2d_batch_norm2d_forwardV2 n:%d c_out:%d h_out:%d w_out:%d output[%d]: %f\n", n, c_out, h_in/stride, w_in/stride, offset_out, value);
//                     // }
//                 }
//             }
//         }
//     }
// #ifdef UNIAD_FUSE_CONV2D_BATCH_NORM2D_DEBUG
//     printf("fuse_conv2d_batch_norm2d_fwd: (%d, %d, %d, %d) padding: %d\n", N, C_out, H_out, W_out, padding);
//     printf("[");
//     for (int n = 0; n < N; n++) {
//         printf("[");
//         for (int c = 0; c < C_out; c++) {
//             printf("[");
//             // for (int i = 0; i < out_height; i++) {
//             for (int h = 0; h < 16; h++) {
//                 printf("[");
//                 // for (int j = 0; j < out_width; j++) {
//                 for (int w = 0; w < 16; w++) {
//                     int offset = n * C_out * H_out * W_out
//                                    + c * H_out * W_out
//                                    + h * W_out
//                                    + w;
//                     printf("%.3f, ", output[offset]);
//                 }
//                 printf("],\n");
//             }
//             printf("],\n");
//         }
//         printf("],\n");
//     }
//     printf("]\n");
// #endif
// }

void fuse_conv2d_batch_norm2d_fwd(float * output, int * output_shape, float * input, int * input_shape,
                float * filters, int * filters_shape, float * bias, 
                int * strides, int * paddings, const char * padding_mode,
                const float * gamma, const float * beta, const float * mean, const float * var) {
#if defined(UNIAD_BENCHMARK) || defined(UNIAD_FUSE_CONV2D_BATCH_NORM2D_BENCHMARK)
    double tdata = omp_get_wtime();
#endif // UNIAD_BENCHMARK
    cc_fuse_conv2d_batch_norm2d_fwd(output, output_shape, input, input_shape, filters, filters_shape, bias, strides, paddings, padding_mode,
                                    gamma, beta, mean, var);
#if defined(UNIAD_BENCHMARK) || defined(UNIAD_FUSE_CONV2D_BATCH_NORM2D_BENCHMARK)
    tdata = omp_get_wtime() - tdata;
    printf("[benchmark][fuse_conv2d_batch_norm2d_fwd][cc_fuse_conv2d_batch_norm2d_fwd]: in_shape:(%d, %d, %d, %d),  weight_shape:(%d, %d, %d, %d), out_shape:(%d, %d, %d, %d), "
           "strides:(%d, %d), panddings:(%d, %d) in %f secs\n", 
           input_shape[0], input_shape[1], input_shape[2], input_shape[3], 
           filters_shape[0], filters_shape[1], filters_shape[2], filters_shape[3],
           output_shape[0], output_shape[1], output_shape[2], output_shape[3], 
           strides[0], strides[1], paddings[0], paddings[1], tdata);
#endif // UNIAD_BENCHMARK
}

#endif // UNIAD_FUSE_CONV2D_BATCH_NORM2D_DEBUG