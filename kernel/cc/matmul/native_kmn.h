#ifndef __LD_UNIAD_NATIVE_KMN_MATMUL_H__
#define __LD_UNIAD_NATIVE_KMN_MATMUL_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>


void cc_native_kmn_matmul(float * C, const float * A, float * B, const int M, const int N, const int K) {
    for (int k = 0; k < K; k++) {
        #pragma omp parallel for 
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                int C_offset = i * N + j;
                int A_offset = i * K + k;
                int B_offset = j * K + k;
                C[C_offset] += A[A_offset] * B[B_offset];
            }
        }
    }
#ifdef UNIAD_NATIVE_KMN_MATMUL_DEBUG
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
#endif // UNIAD_NATIVE_KMN_MATMUL_DEBUG
}


double native_kmn_matmul(float * C, const float * A, float * B, const int M, const int N, const int K) {
    struct timeval start, end;
    double time_used;
    gettimeofday(&start,NULL);

    cc_native_kmn_matmul(C, A, B, M, N, K);

    gettimeofday(&end,NULL);
    time_used = end.tv_sec-start.tv_sec + (end.tv_usec-start.tv_usec)/1000000.0;
#if defined(UNIAD_BENCHMARK) || defined(UNIAD_NATIVE_KMN_MATMUL_BENCHMARK)
    printf("[benchmark][native_kmn_matmul][cc_native_kmn_matmul]: MNK:(%d, %d, %d), in %f secs\n", M, N, K, time_used);
#endif // UNIAD_BENCHMARK
    return time_used;
}

#endif // __LD_UNIAD_NATIVE_KMN_MATMUL_H__