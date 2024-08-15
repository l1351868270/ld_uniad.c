#ifndef __LD_UNIAD_KITTENS_GEMM_128_H__
#define __LD_UNIAD_KITTENS_GEMM_128_H__

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include "kittens.cuh"

namespace bench {
namespace kittens_gemm_128 {

constexpr int NUM_WORKERS = 8;
constexpr int MMA_M = kittens::TILE_DIM;
constexpr int default_alignment = 1024;

using st_hf_1x8_a = kittens::st_hf<1, 8, kittens::ducks::st_layout::swizzle>;
using st_hf_1x8_b = kittens::st_hf<1, 8, kittens::ducks::st_layout::swizzle>;
using st_hf_1x1_c = kittens::st_hf<1, 1, kittens::ducks::st_layout::swizzle>;

using rt_hf_1x8_a = kittens::rt_hf<1, 8, kittens::ducks::rt_layout::row>;
using rt_hf_1x8_b = kittens::rt_hf<1, 8, kittens::ducks::rt_layout::row>;
using rt_hf_1x1_c = kittens::rt_hf<1, 1, kittens::ducks::rt_layout::row>;

// template <typename T>
__global__ void cuda_kittens_gemm_kernel_128(half * C, half * A, half * B, const int M, const int N, const int K) {
    auto warpid = kittens::warpid();
    auto laneid = kittens::laneid();
    const size_t warp_row = blockIdx.x * MMA_M * NUM_WORKERS + warpid * kittens::TILE_DIM;

    extern __shared__ kittens::alignment_dummy __shm[];
    kittens::shared_allocator<default_alignment> al((int*)&__shm[0]);
    st_hf_1x8_a (&sA)[NUM_WORKERS] = al.allocate<st_hf_1x8_a, NUM_WORKERS>();
    st_hf_1x8_b (&sB)[NUM_WORKERS] = al.allocate<st_hf_1x8_b, NUM_WORKERS>();
    st_hf_1x1_c (&sC)[NUM_WORKERS] = al.allocate<st_hf_1x1_c, NUM_WORKERS>();

    rt_hf_1x8_a afrag;
    rt_hf_1x8_b bfrag;
    rt_hf_1x1_c cfrag;

    kittens::zero(cfrag);


    kittens::load(sA[warpid], A + warp_row * K, K);
    if (blockIdx.x == 0) {
        kittens::load(sB[warpid], B + warpid * MMA_M * K, K);
    }
    
    
    if (warp_row >= M) {
        return;
    }
    
    kittens::load(afrag, sA[warpid]);

    kittens::load(bfrag, sB[warpid]);
    kittens::mma_ABt(cfrag, afrag, bfrag, cfrag);


    for (int j = 0; j < 8; j++) {
        kittens::store(C + warp_row * N + j * kittens::TILE_DIM, cfrag, N);
    }
    // kittens::store(C + warp_row * N + warp_col, cfrag, N);

}

void launch_kittens_gemm(half * C, half * A, half * B, const int M, const int N, const int K) {
    // row-major
    const dim3 block_dim{32 * NUM_WORKERS};
    const dim3 grid_dim{(unsigned int)(M + MMA_M - 1) / MMA_M};
    int smem_size = 2 * sizeof(st_hf_1x8_a)
                  + 8 *sizeof(st_hf_1x8_b)
                  + 8 * sizeof(st_hf_1x1_c)
                  + 2 * default_alignment;
    // printf("[benchmark][kittens_gemm_128][cuda_kittens_gemm] smem_size: %d\n", smem_size);
    cudaFuncSetAttribute(cuda_kittens_gemm_kernel_128, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    cuda_kittens_gemm_kernel_128<<<grid_dim, block_dim, smem_size>>>(C, A, B, M, N, K);
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
    printf("[benchmark][kittens_gemm_128][cuda_kittens_gemm]: MNK:(%d, %d, %d), in %f ms\n", M, N, K, time_used);
#endif // UNIAD_BENCHMARK
    return time_used;
}

#endif // __LD_UNIAD_KITTENS_GEMM_128_H__
} // namespace bench
} // namespace kittens_gemm_128