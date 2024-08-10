#ifndef __LD_UNIAD_KITTENS_GEMM_H__
#define __LD_UNIAD_KITTENS_GEMM_H__

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "kittens.cuh"

namespace bench {
namespace kittens_gemm {

constexpr int MMA_M = 16;
constexpr int MMA_N = 16;
constexpr int MMA_K = 16;

__global__ void cuda_kittens_gemm_kernel(half * C, half * A, half * B, const int M, const int N, const int K) {
    const size_t K_tiles = (K + MMA_K - 1) / MMA_K;
    const size_t warp_row = blockIdx.y * MMA_M;
    const size_t warp_col = blockIdx.x * MMA_N;
    kittens::rt_hf_1x1<> cr;
    kittens::zero(cr);

    #pragma unroll
    for (size_t i = 0; i < K_tiles; i++) {
        kittens::rt_hf_1x1<> ar;
        kittens::rt_hf_1x1<> br;

        kittens::load(ar, A + warp_row * K + i * MMA_K, K);
        kittens::load(br, B + warp_col * K + i * MMA_K, K);

        kittens::mma_ABt(cr, ar, br, cr);
    }
    kittens::store(C + warp_row * N + warp_col, cr, N);
    if (warp_row >= M && warp_col >= N) {
        return;
    }

}

void launch_kittens_gemm(half * C, half * A, half * B, const int M, const int N, const int K) {
    // row-major
    const dim3 block_dim{32u};
    const dim3 grid_dim{(unsigned int)(N + MMA_N - 1) / MMA_N,
                        (unsigned int)(M + MMA_M - 1) / MMA_M};

    cuda_kittens_gemm_kernel<<<grid_dim, block_dim>>>(C, A, B, M, N, K);
}


double kittens_gemm(half * C, half * A, half * B, const int M, const int N, const int K) {
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);

    launch_kittens_gemm(C, A, B, M, N, K);
 
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time_used = 0.0;
    cudaEventElapsedTime(&time_used, start, end);
#if defined(UNIAD_BENCHMARK) || defined(UNIAD_BLAS_MATMUL_BENCHMARK)
    printf("[benchmark][kittens_gemm][cuda_kittens_gemm]: MNK:(%d, %d, %d), in %f ms\n", M, N, K, time_used);
#endif // UNIAD_BENCHMARK
    return time_used;
}

#endif // __LD_UNIAD_KITTENS_GEMM_H__
} // namespace bench
} // namespace kittens_gemm