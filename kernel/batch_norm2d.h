#ifndef __LD_UNIAD_BATCH_NORM2D_H__
#define __LD_UNIAD_BATCH_NORM2D_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
void cc_batch_norm2d_fwd(float * dst, const float * src, const float * gamma, const float * beta, const float * mean, const float * var, 
                      const int N, const int C, const int H, const int W) {
    #pragma omp parallel for collapse(4)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    int offset = n * C * H * W + c * H * W + h * W + w;
                    dst[offset] = (src[offset] - mean[c]) / sqrtf(var[c] + 1e-5) * gamma[c] + beta[c];
                }
            }
        }
    }
#ifdef UNIAD_BATCH_NORM2D_DEBUG
    printf("batch_norm2d_fwd: (%d, %d, %d, %d)\n", N, C, H, W);
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
#endif // UNIAD_BATCH_NORM2D_DEBUG
}

void batch_norm2d_fwd(float * dst, const float * src, const float * gamma, const float * beta, const float * mean, const float * var, 
                      const int N, const int C, const int H, const int W) {
#if defined(UNIAD_BENCHMARK) || defined(UNIAD_BATCH_NORM2D_BENCHMARK)
    double tdata = omp_get_wtime();
#endif // UNIAD_BENCHMARK
    cc_batch_norm2d_fwd(dst, src, gamma, beta, mean, var, N, C, H, W);
#if defined(UNIAD_BENCHMARK) || defined(UNIAD_BATCH_NORM2D_BENCHMARK)
    tdata = omp_get_wtime() - tdata;
    printf("[benchmark][batch_norm2d_fwd][cc_batch_norm2d_fwd]: NCHW:(%d, %d, %d, %d), in %f secs\n", N, C, H, W, tdata);
#endif // UNIAD_BENCHMARK
}

#endif // __LD_UNIAD_BATCH_NORM2D_H__