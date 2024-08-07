#ifndef __LD_UNIAD_BLAS_MATMUL_H__
#define __LD_UNIAD_BLAS_MATMUL_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>
#include <cblas.h>

void cc_blas_matmul(float * C, const float * A, float * B, const int M, const int N, const int K) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0, A, K, B, K, 0.0, C, N);
#ifdef UNIAD_BLAS_MATMUL_DEBUG
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
#endif // UNIAD_BLAS_MATMUL_DEBUG
}


double blas_matmul(float * C, const float * A, float * B, const int M, const int N, const int K) {
    struct timeval start, end;
    double time_used;
    gettimeofday(&start,NULL);

    cc_blas_matmul(C, A, B, M, N, K);

    gettimeofday(&end,NULL);
    time_used = end.tv_sec-start.tv_sec + (end.tv_usec-start.tv_usec)/1000000.0;
#if defined(UNIAD_BENCHMARK) || defined(UNIAD_BLAS_MATMUL_BENCHMARK)
    printf("[benchmark][blas_matmul][cc_blas_matmul]: MNK:(%d, %d, %d), in %f secs\n", M, N, K, time_used);
#endif // UNIAD_BENCHMARK
    return time_used;
}

#endif // __LD_UNIAD_BLAS_MATMUL_H__