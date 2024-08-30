#ifndef __LD_UNIAD_BENCH_IMPLICIT_GEMM_H__
#define __LD_UNIAD_BENCH_IMPLICIT_GEMM_H__

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include <string.h>
namespace bench {
namespace cc_implicit_gemm {
// https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
// https://github.com/NVIDIA/cutlass/blob/main/media/docs/implicit_gemm_convolution.md#implicit-gemm-algorithm
void cc_conv2d_fwd(float * y_ptr, float * x_ptr, float * w_ptr, int N, int C, int H, int W,
                   int K, int P, int Q, int R, int S, int U, int V, 
                   int pad_h, int pad_w, int dila_h, int dila_w, const char * padding_mode) {
    if (strcmp(padding_mode, "zeros")) {
        fprintf(stderr, "[cc_conv2d_fwd]: padding_mode only suppoer zeros\n");
    }
    int GEMM_M = N * P * Q;
    int GEMM_N = K;
    int GEMM_K = C * R * S;
    // printf("GEMM_M: %d, GEMM_N: %d, GEMM_K: %d\n", GEMM_M, GEMM_N, GEMM_K);
    // printf("N: %d, C: %d, H: %d, W: %d, K: %d, P: %d, Q: %d, R: %d, S: %d, U: %d, V: %d, pad_h: %d, pad_w: %d, dila_h: %d, dila_w: %d\n", 
    //        N, C, H, W, K, P, Q, R, S, U, V, pad_h, pad_w, dila_h, dila_w);
    #pragma omp parallel for collapse(2)
    for (int gemm_i = 0; gemm_i < GEMM_M; ++gemm_i) {
        for (int gemm_j = 0; gemm_j < GEMM_N; ++gemm_j) {
            int n = gemm_i / (P * Q);
            int npq_residual = gemm_i % (P * Q);
            int p = npq_residual / Q;
            int q = npq_residual % Q;
            float accum = 0.0f;
            for (int gemm_k = 0; gemm_k < GEMM_K; ++gemm_k) {
                int k = gemm_j;

                int r = gemm_k / (S * C);
                int rsc_residual = gemm_k % (S * C);

                int s = rsc_residual / C;
                int c = rsc_residual % C;

                int h = p * U + r * dila_h - pad_h;
                int w = q * V + s * dila_w - pad_w;
                int offs_a = n * H * W * C + h * W * C + w * C + c;
                int offs_w = k * R * S * C + r * S * C + s * C + c;
                
                if (h >= 0 && h < H && w >= 0 && w < W) {
                    // printf("n:%d, p:%d, q:%d, k:%d, r:%d, s:%d, c:%d, h:%d, w:%d, dila_h:%d, dila_w:%d, a[%d,%d]:%f, b[%d, %d]:%f\n", 
                    //    n, p, q, k, r, s, c, h, w, dila_h, dila_w,
                    //    gemm_i, gemm_k, x_ptr[offs_a], gemm_k, gemm_j, w_ptr[offs_w]);
                    accum += x_ptr[offs_a] * w_ptr[offs_w];
                    
                }
            }
            
            int offs_y = gemm_i * GEMM_N + gemm_j;
            y_ptr[offs_y] = accum;
        }
    }
}

double conv2d_fwd(float * y_ptr, float * x_ptr, float * w_ptr, int N, int C, int H, int W,
                   int K, int P, int Q, int R, int S, int U, int V, 
                   int pad_h, int pad_w, int dila_h, int dila_w, const char * padding_mode){
    struct timeval start, end;
    double time_used;
    gettimeofday(&start,NULL);

    cc_conv2d_fwd(y_ptr, x_ptr, w_ptr, N, C, H, W, K, P, Q, R, S, U, V, pad_h, pad_w, dila_h, dila_w, padding_mode);
    
    gettimeofday(&end,NULL);
    time_used = end.tv_sec-start.tv_sec + (end.tv_usec-start.tv_usec)/1000000.0;
    time_used *= 1000.0; // ms
    return time_used;
}
} // namespace cc_implicit_gemm
} // namespace bench
#endif // UNIAD_BENCH_IMPLICIT_GEMM_DEBUG