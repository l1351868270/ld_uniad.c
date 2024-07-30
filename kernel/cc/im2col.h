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

    // #pragma omp parallel for collapse(4)
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


void cc_im2col_v1_fwd(float * data_col, const float * data_im, const int N, const int C, const int H, const int W, 
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

    // #pragma omp parallel for collapse(4)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H + 2 * pad_h; h += stride_h) {
                for (int w = 0; w < W + 2 * pad_w; w += stride_w) {
                    float in_val = 0.0f;
                    if (h >= pad_h && h - pad_h < H && w >= pad_w && w - pad_w < W) {
                        in_val = data_im[n * C * H * W + c * H * W + (h - pad_h) * W + (w - pad_w)];
                    }
                    for (int i = 0; i < ksize_h; i++) {
                        for (int j = 0; j < ksize_w; j++) {
                            if (h - i >= 0 && w - j >=0) {
                                int h_out = (h - i) / stride_h;
                                int w_out = (w - j) / stride_w;
                                if (h_out < H_out && w_out < W_out) {
                                    int im2col_offset = n * H_out * W_out * im2col_cols 
                                                      + (h_out * W_out + w_out) * im2col_cols
                                                      + c * ksize_h * ksize_w 
                                                      + i * ksize_w + j;
                                    data_col[im2col_offset] = in_val;
                                    // printf("im2col_offset: %d, in_val: %.3f, h_out: %d, (%d, %d, %d, %d, %d, %d)\n", im2col_offset, in_val, h_out, n, c, h, w, i, j);
                                }
                            }

                            // int im_h = h_out * stride_h - pad_h + i;
                            // int im_w = w_out * stride_w - pad_w + j;
                            // int im_offset = n * C * H * W 
                            //               + c * H * W 
                            //               + im_h * W 
                            //               + im_w;
                            // int col_offset = n * H_out * W_out * im2col_cols 
                            //                + (h_out * W_out + w_out) * im2col_cols 
                            //                + c * ksize_h * ksize_w 
                            //                + i * ksize_w 
                            //                + j;
                            // if (im_h >= 0 && im_h < H && im_w >= 0 && im_w < W) {
                            //     data_col[col_offset] = data_im[im_offset];
                            // } 
                            // else {
                            //     data_col[col_offset] = 0.0f;
                            // }
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

void cc_im2col_v2_fwd(float * data_col, const float * data_im, const int N, const int C, const int H, const int W, 
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

    // #pragma omp parallel for collapse(1)
    for (int n = 0; n < N; n++) {
        for (int h_out = 0; h_out < H_out; h_out++) {
            for (int w_out = 0; w_out < W_out; w_out++) {
                for (int c = 0; c < C; c++) {
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

// float im2col_get_pixel(float *im, int height, int width, int channels,
//                         int row, int col, int channel, int pad)
// {
//     row -= pad;
//     col -= pad;

//     if (row < 0 || col < 0 ||
//         row >= height || col >= width) return 0;
//     return im[col + width*(row + height*channel)];
// }

// void im2col_cpu(float* data_im,
//      int channels,  int height,  int width,
//      int ksize,  int stride, int pad, float* data_col) 
// {
//     int c,h,w;
//     int height_col = height;
//     int width_col = width;

//     int channels_col = channels * ksize * ksize;
//     for (c = 0; c < channels_col; ++c) {
//         int w_offset = c % ksize;
//         int h_offset = (c / ksize) % ksize;
//         int c_im = c / ksize / ksize;
//         for (h = 0; h < height_col; ++h) {
//             for (w = 0; w < width_col; ++w) {
//                 int im_row = h_offset + h * stride;
//                 int im_col = w_offset + w * stride;
//                 int col_index = (c * height_col + h) * width_col + w;
//                 data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
//                         im_row, im_col, c_im, pad);
//             }
//         }
//     }
// }


void im2col_fwd(float * data_col, const float * data_im, const int N, const int C, const int H, const int W, 
                const int * ksizes, const int * strides, const int * paddings, const int * dilations, 
                const char * padding_mode) {
#if defined(UNIAD_BENCHMARK) || defined(UNIAD_IM2COL_BENCHMARK)
    clock_t start, end;
    double time_used;
    start = clock();
#endif // UNIAD_BENCHMARK
    cc_im2col_fwd(data_col, data_im, N, C, H, W, ksizes, strides, paddings, dilations, padding_mode);
#if defined(UNIAD_BENCHMARK) || defined(UNIAD_IM2COL_BENCHMARK)
    end = clock();
    time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("[benchmark][im2col_fwd][cc_im2col_fwd]: NCHW:(%d, %d, %d, %d), in %f secs\n", N, C, H, W, time_used);
#endif // UNIAD_BENCHMARK
}

void im2col_v1_fwd(float * data_col, const float * data_im, const int N, const int C, const int H, const int W, 
                const int * ksizes, const int * strides, const int * paddings, const int * dilations, 
                const char * padding_mode) {
#if defined(UNIAD_BENCHMARK) || defined(UNIAD_IM2COL_BENCHMARK)
    clock_t start, end;
    double time_used;
    start = clock();
#endif // UNIAD_BENCHMARK
    cc_im2col_v1_fwd(data_col, data_im, N, C, H, W, ksizes, strides, paddings, dilations, padding_mode);
#if defined(UNIAD_BENCHMARK) || defined(UNIAD_IM2COL_BENCHMARK)
    end = clock();
    time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("[benchmark][im2col_v1_fwd][cc_im2col_v1_fwd]: NCHW:(%d, %d, %d, %d), in %f secs\n", N, C, H, W, time_used);
#endif // UNIAD_BENCHMARK
}

void im2col_v2_fwd(float * data_col, const float * data_im, const int N, const int C, const int H, const int W, 
                const int * ksizes, const int * strides, const int * paddings, const int * dilations, 
                const char * padding_mode) {
#if defined(UNIAD_BENCHMARK) || defined(UNIAD_IM2COL_BENCHMARK)
    clock_t start, end;
    double time_used;
    start = clock();
#endif // UNIAD_BENCHMARK
    cc_im2col_v2_fwd(data_col, data_im, N, C, H, W, ksizes, strides, paddings, dilations, padding_mode);
#if defined(UNIAD_BENCHMARK) || defined(UNIAD_IM2COL_BENCHMARK)
    end = clock();
    time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("[benchmark][im2col_v2_fwd][cc_im2col_v2_fwd]: NCHW:(%d, %d, %d, %d), in %f secs\n", N, C, H, W, time_used);
#endif // UNIAD_BENCHMARK
}

#endif // __LD_UNIAD_IM2COL_H__