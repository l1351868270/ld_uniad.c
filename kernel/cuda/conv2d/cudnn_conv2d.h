#ifndef __LD_UNIAD_CUDNN_CONV2D_H__
#define __LD_UNIAD_CUDNN_CONV2D_H__

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cudnn.h>

template <typename T>
void cuda_cudnn_conv2d(cudnnHandle_t * handle,
                       const void *alpha,
                       const cudnnTensorDescriptor_t * xDesc,
                        const void *x,
                        const cudnnFilterDescriptor_t * wDesc,
                        const void *w,
                        const cudnnConvolutionDescriptor_t * convDesc,
                        const void *beta,
                        const cudnnTensorDescriptor_t * yDesc,
                        void *y) {

    if (std::is_same<T, float>::value) {
        // cublasSgemmEx(*handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F, K, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N);
    } else if (std::is_same<T, half>::value) {
        cudnnStatus_t err;
        err = cudnnConvolutionForward(*handle, alpha, 
                                *xDesc, x, 
                                *wDesc, w, 
                                *convDesc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, nullptr, 0, beta, 
                                *yDesc, y);
        if (err != CUDNN_STATUS_SUCCESS) {
            printf("cudnnConvolutionForward failed: %s\n", cudnnGetErrorString(err));
            exit(1);
        }
        // printf("%d: %s\n", err, cudnnGetErrorString(err));
        // cublasSgemmEx(*handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F, K, A, CUDA_R_16F, K, &beta, C, CUDA_R_16F, N);
    } else if (std::is_same<T, nv_bfloat16>::value) {
        // cublasSgemmEx(*handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16BF, K, A, CUDA_R_16BF, K, &beta, C, CUDA_R_16BF, N);
    } else {
        printf("Unsupported data type for cuda_blas_matmul\n");
        exit(1);
    }
    
     
   
// #ifdef UNIAD_CUDNN_CONV2D_DEBUG
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
// #endif // UNIAD_CUDNN_CONV2D_DEBUG
}

template <typename T>
double cudnn_conv2d(T * y, T * x, const T * w, int N, int H, int W, int C, 
                    int K, int R, int S, int pad_h, int pad_w, int U, int V, int dilation_h, int dilation_w) {


    cudnnHandle_t handle;
    cudnnCreate(&handle);
    cudnnTensorDescriptor_t xDesc;
    cudnnCreateTensorDescriptor(&xDesc);
    cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, N, C, H, W);
    cudnnFilterDescriptor_t wDesc;
    cudnnCreateFilterDescriptor(&wDesc);
    cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_HALF, CUDNN_TENSOR_NHWC, K, C, R, S);
    cudnnConvolutionDescriptor_t convDesc;
    cudnnCreateConvolutionDescriptor(&convDesc);
    cudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w, U, V, dilation_h, dilation_w, CUDNN_CONVOLUTION, CUDNN_DATA_HALF);

    int batch_size{0}, channels{0}, height{0}, width{0};
    cudnnGetConvolution2dForwardOutputDim(convDesc,
                                                   xDesc,
                                                   wDesc,
                                                   &batch_size,
                                                   &channels,
                                                   &height,
                                                   &width);

    // printf("batch_size: %d, channels: %d, height: %d, width: %d\n", batch_size, channels, height, width);
    cudnnTensorDescriptor_t yDesc;
    cudnnCreateTensorDescriptor(&yDesc);
    int P = floor((H + 2 * pad_h - dilation_h * (R - 1) - 1) / U + 1);
    int Q = floor((W + 2 * pad_w - dilation_w * (S - 1) - 1) / V + 1);
    cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, N, K, P, Q);

    const float alpha = 1.0;
    const float beta = 0.0;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);

    cuda_cudnn_conv2d<T>(&handle, &alpha, &xDesc, x, &wDesc, w, &convDesc, &beta, &yDesc, y);
    // cudnnDeviceSynchronize();
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time_used = 0.0;
    cudaEventElapsedTime(&time_used, start, end);

    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroyFilterDescriptor(wDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroyTensorDescriptor(yDesc);
    cudnnDestroy(handle);

#if defined(UNIAD_BENCHMARK) || defined(UNIAD_CUDNN_CONV2D_BENCHMARK)
    printf("[benchmark][blas_matmul][cuda_blas_matmul]: MNK:(%d, %d, %d), in %f ms\n", M, N, K, time_used);
#endif // UNIAD_BENCHMARK
    return time_used;
}

#endif // __LD_UNIAD_CUDNN_CONV2D_H__