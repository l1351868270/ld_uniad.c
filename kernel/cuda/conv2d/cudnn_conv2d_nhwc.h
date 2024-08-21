#ifndef __LD_UNIAD_CUDNN_CONV2D_NHWC_H__
#define __LD_UNIAD_CUDNN_CONV2D_NHWC_H__

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cudnn.h>
// #include <cudnn_frontend_find_plan.h>
// #include <cudnn_frontend_get_plan.h>

namespace bench {
namespace cudnn_conv2d_nhwc {

template <typename T>
void cuda_cudnn_conv2d_nhwc(cudnnHandle_t handle,
                       const void * alpha,
                       const cudnnTensorDescriptor_t xDesc,
                        const void *x,
                        const cudnnFilterDescriptor_t wDesc,
                        const void *w,
                        const cudnnConvolutionDescriptor_t convDesc,
                        cudnnConvolutionFwdAlgo_t algo,
                        void *workSpace,
                        size_t workSpaceSizeInBytes,
                        const void * beta,
                        const cudnnTensorDescriptor_t yDesc,
                        void *y) {
    // cudnn_frontend::VariantPackBuilder();
    if (std::is_same<T, float>::value) {
        // cublasSgemmEx(*handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F, K, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N);
    } else if (std::is_same<T, half>::value) {
        cudnnStatus_t err;
        err = cudnnConvolutionForward(handle, alpha, 
                                xDesc, x, 
                                wDesc, w, 
                                convDesc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, workSpace, workSpaceSizeInBytes, beta, 
                                yDesc, y);
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

    cudnnStatus_t err;
    cudnnHandle_t handle;
    err = cudnnCreate(&handle);
    if (err != CUDNN_STATUS_SUCCESS) {
        printf("cudnnCreate failed: %s\n", cudnnGetErrorString(err));
        exit(1);
    }
    cudnnTensorDescriptor_t xDesc;
    err = cudnnCreateTensorDescriptor(&xDesc);
    if (err != CUDNN_STATUS_SUCCESS) {
        printf("cudnnCreateTensorDescriptor failed: %s\n", cudnnGetErrorString(err));
        exit(1);
    }
    // err = cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, N, C, H, W);
    // if (err != CUDNN_STATUS_SUCCESS) {
    //     printf("cudnnSetTensor4dDescriptor failed: %s\n", cudnnGetErrorString(err));
    //     exit(1);
    // }

    err = cudnnSetTensor4dDescriptorEx(xDesc, CUDNN_DATA_HALF, N, C, H, W, H * W * C, 1, W * C, C);
    if (err != CUDNN_STATUS_SUCCESS) {
        printf("cudnnSetTensor4dDescriptorEx failed: %s\n", cudnnGetErrorString(err));
        exit(1);
    }

    cudnnFilterDescriptor_t wDesc;
    err = cudnnCreateFilterDescriptor(&wDesc);
    if (err != CUDNN_STATUS_SUCCESS) {
        printf("cudnnCreateFilterDescriptor failed: %s\n", cudnnGetErrorString(err));
        exit(1);
    }
    err = cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_HALF, CUDNN_TENSOR_NHWC, K, C, R, S);
    if (err != CUDNN_STATUS_SUCCESS) {
        printf("cudnnSetFilter4dDescriptor failed: %s\n", cudnnGetErrorString(err));
        exit(1);
    }
    cudnnConvolutionDescriptor_t convDesc;
    err = cudnnCreateConvolutionDescriptor(&convDesc);
    if (err != CUDNN_STATUS_SUCCESS) {
        printf("cudnnCreateConvolutionDescriptor failed: %s\n", cudnnGetErrorString(err));
        exit(1);
    }

    err = cudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w, U, V, dilation_h, dilation_w, CUDNN_CROSS_CORRELATION, CUDNN_DATA_HALF);
    if (err != CUDNN_STATUS_SUCCESS) {
        printf("cudnnSetConvolution2dDescriptor failed: %s\n", cudnnGetErrorString(err));
        exit(1);
    }

    
    int batch_size{0}, channels{0}, height{0}, width{0};
    err = cudnnGetConvolution2dForwardOutputDim(convDesc,
                                                   xDesc,
                                                   wDesc,
                                                   &batch_size,
                                                   &channels,
                                                   &height,
                                                   &width);
    if (err != CUDNN_STATUS_SUCCESS) {
        printf("cudnnGetConvolution2dForwardOutputDim failed: %s\n", cudnnGetErrorString(err));
        exit(1);
    }
    // cudnnGetConvolutionForwardAlgorithm_v7();
    // printf("batch_size: %d, channels: %d, height: %d, width: %d\n", batch_size, channels, height, width);
    cudnnTensorDescriptor_t yDesc;
    err = cudnnCreateTensorDescriptor(&yDesc);
    if (err != CUDNN_STATUS_SUCCESS) {
        printf("cudnnCreateTensorDescriptor failed: %s\n", cudnnGetErrorString(err));
        exit(1);
    }
    int P = floor((H + 2 * pad_h - dilation_h * (R - 1) - 1) / U + 1);
    int Q = floor((W + 2 * pad_w - dilation_w * (S - 1) - 1) / V + 1);
    // err = cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, N, K, P, Q);
    // if (err != CUDNN_STATUS_SUCCESS) {
    //     printf("cudnnSetTensor4dDescriptor failed: %s\n", cudnnGetErrorString(err));
    //     exit(1);
    // }
    err = cudnnSetTensor4dDescriptorEx(yDesc, CUDNN_DATA_HALF, N, K, P, Q, P * Q * K, 1, Q * K, K);
    if (err != CUDNN_STATUS_SUCCESS) {
        printf("cudnnSetTensor4dDescriptorEx failed: %s\n", cudnnGetErrorString(err));
        exit(1);
    }

    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    void *workSpace;
    size_t workSpaceSizeInBytes;
    err = cudnnGetConvolutionForwardWorkspaceSize(handle, xDesc, wDesc, convDesc, yDesc, algo, &workSpaceSizeInBytes);
    if (err != CUDNN_STATUS_SUCCESS) {
        printf("cudnnGetConvolutionForwardWorkspaceSize failed: %s\n", cudnnGetErrorString(err));
        exit(1);
    }
    printf("workSpaceSizeInBytes: %ld\n", workSpaceSizeInBytes);
    cudaMalloc(&workSpace, workSpaceSizeInBytes);
    const float alpha = 1.0;
    const float beta = 0.0;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);

    cuda_cudnn_conv2d_nhwc<T>(handle, &alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, &beta, yDesc, y);
    // cudnnDeviceSynchronize();
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time_used = 0.0;
    cudaEventElapsedTime(&time_used, start, end);


    err = cudnnDestroyTensorDescriptor(xDesc);
    if (err != CUDNN_STATUS_SUCCESS) {
        printf("cudnnDestroyTensorDescriptor failed: %s\n", cudnnGetErrorString(err));
        exit(1);
    }
    err = cudnnDestroyFilterDescriptor(wDesc);
    if (err != CUDNN_STATUS_SUCCESS) {
        printf("cudnnDestroyFilterDescriptor failed: %s\n", cudnnGetErrorString(err));
        exit(1);
    }
    err = cudnnDestroyConvolutionDescriptor(convDesc);
    if (err != CUDNN_STATUS_SUCCESS) {
        printf("cudnnDestroyConvolutionDescriptor failed: %s\n", cudnnGetErrorString(err));
        exit(1);
    }
    err = cudnnDestroyTensorDescriptor(yDesc);
    if (err != CUDNN_STATUS_SUCCESS) {
        printf("cudnnDestroyTensorDescriptor failed: %s\n", cudnnGetErrorString(err));
        exit(1);
    }
    err = cudnnDestroy(handle);
    if (err != CUDNN_STATUS_SUCCESS) {
        printf("cudnnDestroy failed: %s\n", cudnnGetErrorString(err));
        exit(1);
    }

    cudaFree(workSpace);

    return time_used;
}

} // namespace cudnn_conv2d_nhwc
} // namespace bench

#endif // __LD_UNIAD_CUDNN_CONV2D_NHWC_H__