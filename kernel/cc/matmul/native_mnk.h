#ifndef __LD_UNIAD_NATIVE_MNK_MATMUL_H__
#define __LD_UNIAD_NATIVE_MNK_MATMUL_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>


void cc_native_mnk_matmul(float * C, const float * A, float * B, const int M, const int N, const int K) {
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int C_offset = i * N + j;
            float val = C[C_offset];
            for (int k = 0; k < K; k++) {
                int A_offset = i * K + k;
                int B_offset = j * K + k;
                val += A[A_offset] * B[B_offset];
            }
            C[C_offset] = val;
        }
    }
#ifdef UNIAD_NATIVE_MNK_MATMUL_DEBUG
    printf("cc_matmul_fwd: (%d, %d, %d)\n", M, N, K);
    printf("[");
    for (int i = 0; i < M; i++) {
        printf("[");
        for (int j = 0; j < N; j++) {
            int offset = i * N + j;
            printf("%.3f, ", C[offset]);
        }
        printf("],\n");
    }
    printf("]\n");
#endif // UNIAD_NATIVE_MNK_MATMUL_DEBUG
}


double native_mnk_matmul(float * C, const float * A, float * B, const int M, const int N, const int K) {
    struct timeval start, end;
    double time_used;
    gettimeofday(&start,NULL);

    cc_native_mnk_matmul(C, A, B, M, N, K);

    gettimeofday(&end,NULL);
    time_used = end.tv_sec-start.tv_sec + (end.tv_usec-start.tv_usec)/1000000.0;
#if defined(UNIAD_BENCHMARK) || defined(UNIAD_NATIVE_MNK_MATMUL_BENCHMARK)
    printf("[benchmark][native_mnk_matmul][cc_native_mnk_matmul]: MNK:(%d, %d, %d), in %f secs\n", M, N, K, time_used);
#endif // UNIAD_BENCHMARK
    return time_used;
}

#endif // __LD_UNIAD_NATIVE_MNK_MATMUL_H__