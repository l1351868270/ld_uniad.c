#ifndef __LD_UNIAD_MATMUL_H__
#define __LD_UNIAD_MATMUL_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <cblas.h>

void cc_matmul_fwd(float * C, const float * A, float * B, const int M, const int N, const int K) {
    #pragma omp parallel for collapse(2) num_threads(4)
    for (int i = 0; i < M; i++) {
        // #pragma omp parallel for 
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                    int A_offset = i * K + k;
                    int B_offset = j * K + k;
                    int C_offset = i * N + j;
                    C[C_offset] += A[A_offset] * B[B_offset];
            }
        }
    }
#ifdef UNIAD_MATMUL_DEBUG
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
#endif // UNIAD_MATMUL_DEBUG
}

void openblas_matmul_fwd(float * C, const float * A, float * B, const int M, const int N, const int K) {
    // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0, A, K, B, N, 0.0, C, N);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0, A, K, B, K, 0.0, C, N);
#ifdef UNIAD_MATMUL_DEBUG
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
#endif // UNIAD_MATMUL_DEBUG
}

void matmul_fwd(float * C, const float * A, float * B, const int M, const int N, const int K) {
#if defined(UNIAD_BENCHMARK) || defined(UNIAD_MATMUL_BENCHMARK)
    clock_t start, end;
    double time_used;
    start = clock();
#endif // UNIAD_BENCHMARK
    cc_matmul_fwd(C, A, B, M, N, K);
#if defined(UNIAD_BENCHMARK) || defined(UNIAD_MATMUL_BENCHMARK)
    end = clock();
    time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("[benchmark][matmul_fwd][cc_matmul_fwd]: MNK:(%d, %d, %d), in %f secs\n", M, N, K, time_used);
#endif // UNIAD_BENCHMARK
}

void _openblas_matmul_fwd(float * C, const float * A, float * B, const int M, const int N, const int K) {
#if defined(UNIAD_BENCHMARK) || defined(UNIAD_MATMUL_BENCHMARK)
    clock_t start, end;
    double time_used;
    start = clock();
#endif // UNIAD_BENCHMARK
    openblas_matmul_fwd(C, A, B, M, N, K);
#if defined(UNIAD_BENCHMARK) || defined(UNIAD_MATMUL_BENCHMARK)
    end = clock();
    time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("[benchmark][matmul_fwd][openblas_matmul_fwd]: MNK:(%d, %d, %d), in %f secs\n", M, N, K, time_used);
#endif // UNIAD_BENCHMARK
}

#endif // __LD_UNIAD_MATMUL_H__