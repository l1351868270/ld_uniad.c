#ifndef __LD_UNIAD_KITTENS_GEMM_V3_H__
#define __LD_UNIAD_KITTENS_GEMM_V3_H__

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include "kittens.cuh"

namespace bench {
namespace kittens_gemm_v3 {

constexpr int MMA_M = kittens::TILE_DIM;
constexpr int MMA_N = kittens::TILE_DIM;
constexpr int NUM_K = 4;
constexpr int MMA_K = kittens::TILE_DIM * NUM_K;
constexpr int default_alignment = 1024;

using st_hf_1x4_a = kittens::st_hf<1, NUM_K, kittens::ducks::st_layout::swizzle>;
using st_hf_1x4_b = kittens::st_hf<1, NUM_K, kittens::ducks::st_layout::swizzle>;
using st_hf_1x1_c = kittens::st_hf<1, 1, kittens::ducks::st_layout::swizzle>;

using rt_hf_1x4_a = kittens::rt_hf<1, NUM_K, kittens::ducks::rt_layout::row>;
using rt_hf_1x4_b = kittens::rt_hf<1, NUM_K, kittens::ducks::rt_layout::row>;
using rt_hf_1x1_c = kittens::rt_hf<1, 1, kittens::ducks::rt_layout::row>;

// template <typename T>
__global__ void cuda_kittens_gemm_kernel_v3(half * C, half * A, half * B, const int M, const int N, const int K) {
    auto warpid = kittens::warpid();
    auto laneid = kittens::laneid();
    const size_t warp_row = blockIdx.y * MMA_M;
    const size_t warp_col = blockIdx.x * MMA_N + warpid * kittens::TILE_DIM;

    extern __shared__ kittens::alignment_dummy __shm[];
    kittens::shared_allocator<default_alignment> al((int*)&__shm[0]);
    st_hf_1x4_a (&sA)[2][1] = al.allocate<st_hf_1x4_a, 2, 1>();
    st_hf_1x4_b (&sB)[2][1] = al.allocate<st_hf_1x4_b, 2, 1>();
    st_hf_1x1_c (&sC)[1] = al.allocate<st_hf_1x1_c, 1>();

    rt_hf_1x4_a afrag;
    rt_hf_1x4_b bfrag;
    rt_hf_1x1_c cfrag;
    kittens::zero(cfrag);

    int tic = 0, toc = 1;
    auto block = cooperative_groups::this_thread_block();
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> ab_barrier;
    if (threadIdx.x == 0) {init(&ab_barrier, block.size());}
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> store_barrier;
    if (threadIdx.x == 0) {init(&store_barrier, block.size());}
    block.sync();

    kittens::load_async(sA[tic][0], A + warp_row * K + 0 * MMA_K, K, ab_barrier);
    kittens::load_async(sB[tic][0], B + warp_col * K + 0 * MMA_K, K, ab_barrier);

    // const int a_tile_elements = st_hf_1x4_a::num_elements;
    // const int b_tile_elements = st_hf_1x4_b::num_elements;
    // const int c_tile_elements = st_hf_1x1_c::num_elements;


    auto n_tiles  = K / MMA_K;

    if (warp_row >= M && warp_col >= N) {
        return;
    }
    
    #pragma unroll
    for (size_t i = 0; i < n_tiles;i++, tic ^= 1, toc ^=1) {
        ab_barrier.arrive_and_wait();
        if (i < n_tiles - 1) {
            kittens::load_async(sA[toc][0], A + warp_row * K + (i + 1) * MMA_K, K, ab_barrier);
            kittens::load_async(sB[toc][warpid], B + warp_col * K + (i + 1) * MMA_K, K, ab_barrier);
        }

        kittens::load(afrag, sA[tic][0]);
        kittens::load(bfrag, sB[tic][warpid]);
        kittens::mma_ABt(cfrag, afrag, bfrag, cfrag);
    }

    // kittens::store(sC[warpid], cfrag);
    // block.sync();
    // kittens::store_async(C + warp_row * n + warp_col, sC[warpid], n, store_barrier);
    kittens::store(C + warp_row * N + warp_col, cfrag, N);

    // if (kittens::thread0()) {
    // }
}

void launch_kittens_gemm(half * C, half * A, half * B, const int M, const int N, const int K) {
    // row-major
    const dim3 block_dim{32};
    const dim3 grid_dim{(unsigned int)(N + MMA_N - 1) / MMA_N,
                        (unsigned int)(M + MMA_M - 1) / MMA_M};
    int smem_size = 2 * sizeof(st_hf_1x4_a)
                  + 2 * 1 *sizeof(st_hf_1x4_b)
                  + 1 * sizeof(st_hf_1x1_c)
                  + 2 * default_alignment;
    // printf("[benchmark][kittens_gemm_v3][cuda_kittens_gemm] smem_size: %d\n", smem_size);
    cudaFuncSetAttribute(cuda_kittens_gemm_kernel_v3, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    cuda_kittens_gemm_kernel_v3<<<grid_dim, block_dim, smem_size>>>(C, A, B, M, N, K);
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
    printf("[benchmark][kittens_gemm_v3][cuda_kittens_gemm]: MNK:(%d, %d, %d), in %f ms\n", M, N, K, time_used);
#endif // UNIAD_BENCHMARK
    return time_used;
}

#endif // __LD_UNIAD_KITTENS_GEMM_V3_H__
} // namespace bench
} // namespace kittens_gemm_v3