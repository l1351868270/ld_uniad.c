#ifndef __LD_UNIAD_BLAS_MATMUL_H__
#define __LD_UNIAD_BLAS_MATMUL_H__

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

template <typename T>
void cuda_blas_matmul(cublasHandle_t* handle, T * C, const T * A, T * B, const int M, const int N, const int K) {
    // row-major
    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (std::is_same<T, float>::value) {
        cublasSgemmEx(*handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F, K, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N);
    } else if (std::is_same<T, half>::value) {
        cublasSgemmEx(*handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F, K, A, CUDA_R_16F, K, &beta, C, CUDA_R_16F, N);
    } else if (std::is_same<T, nv_bfloat16>::value) {
        cublasSgemmEx(*handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16BF, K, A, CUDA_R_16BF, K, &beta, C, CUDA_R_16BF, N);
    } else {
        printf("Unsupported data type for cuda_blas_matmul\n");
        exit(1);
    }
    
     
   
// #ifdef UNIAD_BLAS_MATMUL_DEBUG
//     float * h_C = (float *)malloc(M*N*sizeof(float));
//     cudaMemcpy(h_C, C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
//     printf("cuda_matmul_fwd: (%d, %d, %d)\n", M, N, K);
//     printf("[");
//     // for (int i = 0; i < M; i++) {
//     for (int i = 0; i < 16; i++) {
//         printf("[");
//         // for (int j = 0; j < N; j++) {
//         for (int j = 0; j < 16; j++) {
//             int offset = i * N + j;
//             printf("%.3f, ", h_C[offset]);
//         }
//         printf("],\n");
//     }
//     printf("]\n");
// #endif // UNIAD_BLAS_MATMUL_DEBUG
}

template <typename T>
double blas_matmul(cublasHandle_t* handle, T * C, const T * A, T * B, const int M, const int N, const int K) {
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);

    cuda_blas_matmul<T>(handle, C, A, B, M, N, K);
 
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time_used = 0.0;
    cudaEventElapsedTime(&time_used, start, end);
#if defined(UNIAD_BENCHMARK) || defined(UNIAD_BLAS_MATMUL_BENCHMARK)
    printf("[benchmark][blas_matmul][cuda_blas_matmul]: MNK:(%d, %d, %d), in %f ms\n", M, N, K, time_used);
#endif // UNIAD_BENCHMARK
    return time_used;
}

#endif // __LD_UNIAD_BLAS_MATMUL_H__