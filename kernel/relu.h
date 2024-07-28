#ifndef __LD_UNIAD_RELU_H__
#define __LD_UNIAD_RELU_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
void cc_relu_fwd(float * dst, const float * src, const int N, const int C, const int H, const int W) {
    #pragma omp parallel for collapse(4)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    int offset = n * C * H * W + c * H * W + h * W + w;
                    if (src[offset] < 0.0f) {
                        dst[offset] = 0.0f;
                    } else {
                        dst[offset] = src[offset];
                    }
                }
            }
        }
    }
#ifdef UNIAD_RELU_DEBUG
    printf("relu_fwd: (%d, %d, %d, %d)\n", N, C, H, W);
    printf("[");
    for (int n = 0; n < N; n++) {
        printf("[");
        for (int c = 0; c < C; c++) {
            printf("[");
            // for (int h = 0; h < H; h++) {
            for (int h = 0; h < 16; h++) {
                printf("[");
                // for (int w = 0; w < W; w++) {
                for (int w = 0; w < 16; w++) {
                    int offset = n * C * H * W + c * H * W + h * W + w;
                    printf("%.3f, ", dst[offset]);
                }
                printf("],\n");
            }
            printf("],\n");
        }
        printf("],\n");
    }
    printf("]\n");
#endif // UNIAD_RELU_DEBUG
}

void relu_fwd(float * dst, const float * src, const int N, const int C, const int H, const int W) {
#if defined(UNIAD_BENCHMARK) || defined(UNIAD_RELU_BENCHMARK)
    double tdata = omp_get_wtime();
#endif // UNIAD_BENCHMARK
    cc_relu_fwd(dst, src, N, C, H, W);
#if defined(UNIAD_BENCHMARK) || defined(UNIAD_RELU_BENCHMARK)
    tdata = omp_get_wtime() - tdata;
    printf("[benchmark][relu_fwd][cc_relu_fwd]: NCHW:(%d, %d, %d, %d), in %f secs\n", N, C, H, W, tdata);
#endif // UNIAD_BENCHMARK
}

#endif // __LD_UNIAD_RELU_H__