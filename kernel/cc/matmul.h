#ifndef __LD_UNIAD_MATMUL_H__
#define __LD_UNIAD_MATMUL_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>
#include <cblas.h>

void cc_matmul_fwd(float * C, const float * A, float * B, const int M, const int N, const int K) {
    // #pragma omp parallel for collapse(2) num_threads(4)
    for (int i = 0; i < M; i++) {
        // #pragma omp parallel for 
        for (int j = 0; j < N; j++) {
            float val = 0.0f;
            int C_offset = i * N + j;
            for (int k = 0; k < K; k++) {
                int A_offset = i * K + k;
                int B_offset = j * K + k;
                val += A[A_offset] * B[B_offset];
            }
            C[C_offset] = val;
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

void cc_matmul_v1_fwd(float * C, const float * A, float * B, const int M, const int N, const int K) {
    // #pragma omp parallel for collapse(2) num_threads(4)
    for (int k = 0; k < K; k++) {
        for (int i = 0; i < M; i++) {
        // #pragma omp parallel for 
            for (int j = 0; j < N; j++) {
                int C_offset = i * N + j;
                int A_offset = i * K + k;
                int B_offset = j * K + k;
                C[C_offset] += A[A_offset] * B[B_offset];
            }
        }
    }
#ifdef UNIAD_MATMUL_DEBUG
    printf("cc_matmul_v1_fwd: (%d, %d, %d)\n", M, N, K);
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

void blas_matmul_fwd(float * C, const float * A, float * B, const int M, const int N, const int K) {
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
    struct timeval start, end;
    double time_used;
    gettimeofday(&start,NULL);
#endif // UNIAD_BENCHMARK
    cc_matmul_fwd(C, A, B, M, N, K);
#if defined(UNIAD_BENCHMARK) || defined(UNIAD_MATMUL_BENCHMARK)
    gettimeofday(&end,NULL);
    time_used = end.tv_sec-start.tv_sec + (end.tv_usec-start.tv_usec)/1000000.0;
    printf("[benchmark][matmul_fwd][cc_matmul_fwd]: MNK:(%d, %d, %d), in %f secs\n", M, N, K, time_used);
#endif // UNIAD_BENCHMARK
}

void _blas_matmul_fwd(float * C, const float * A, float * B, const int M, const int N, const int K) {
#if defined(UNIAD_BENCHMARK) || defined(UNIAD_MATMUL_BENCHMARK)
    struct timeval start, end;
    double time_used;
    gettimeofday(&start,NULL);
#endif // UNIAD_BENCHMARK
    blas_matmul_fwd(C, A, B, M, N, K);
#if defined(UNIAD_BENCHMARK) || defined(UNIAD_MATMUL_BENCHMARK)
    gettimeofday(&end,NULL);
    time_used = end.tv_sec-start.tv_sec + (end.tv_usec-start.tv_usec)/1000000.0;
    printf("[benchmark][matmul_fwd][blas_matmul_fwd]: MNK:(%d, %d, %d), in %f secs\n", M, N, K, time_used);
#endif // UNIAD_BENCHMARK
}

#endif // __LD_UNIAD_MATMUL_H__