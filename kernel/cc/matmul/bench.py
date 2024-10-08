import time
import os
import numpy as np
import torch
from torch.utils.cpp_extension import load_inline, load
import matplotlib.pyplot as plt

def time_function(func, A, B):
    # Warmup
    for _ in range(1):
        func(A, B)

    start = time.time()
    func(A, B)
    end = time.time()
    return end - start

def avg_time_function(func, A, B, repeat=10):
    # Warmup
    for _ in range(1):
        func(A, B)

    start = time.time()
    for _ in range(repeat):
        func(A, B)
    end = time.time()
    return (end - start) / repeat

def manual_time_function(func, C, A, B):
    # Warmup
    for _ in range(1):
        C.zero_()
        func(C, A, B)
    used_time = func(C, A, B)
    return used_time

def avg_manual_time_function(func, C, A, B, repeat=10):
    # Warmup
    for _ in range(1):
        C.zero_()
        func(C, A, B)
    used_time = 0.0
    for _ in range(repeat):
        C.zero_()
        used_time += func(C, A, B)
    return used_time / repeat

def time_function_v2(func, C, A, B):
    # Warmup
    for _ in range(1):
        func(C, A, B)

    start = time.time()
    func(C, A, B)
    end = time.time()
    return end - start

def torch_matmul(A, B):
    return torch.matmul(A, B)

def np_matmul(A, B):
    return np.matmul(A, B)

def benchmark_matmul():
    # # [2227200, 64, 147],
    # bench_shapes = [[16, 16, 16], [32, 32, 32], [48, 48, 48], [64, 64, 64], [80, 80, 80], [96, 96, 96],
    #                 [112, 112, 112], [128, 128, 128], [144, 144, 144], [160, 160, 160], [176, 176, 176], [192, 192, 192]]
    # bench_shapes = [[64 * i, 64 * i, 64 * i] for i in range(1, 64)]
    # plt_data_x = [f"{m}*{n}*{k}" for m, n, k in bench_shapes]

    M = 2227200
    N = 64
    K = 147
    # M = 2048
    # N = 2048
    # K = 2048
    bench_shapes = [[M, N, K]]

    build_directory = "./kernel/cc/matmul/build/"
    if not os.path.exists(build_directory):
        os.makedirs(build_directory)
    manual_matmul = load(name='manual_matmul', 
                         sources=['kernel/cc/matmul/bench.cpp'], 
                         build_directory=build_directory,
                         verbose=False,
                         extra_include_paths=['kernel/cc/matmul'],
                         extra_cflags=['-O3', '-march=native', '-fopenmp''-std=c++20'],
                         extra_cuda_cflags=['-O3', '-march=native', '-fopenmp', '-std=c++20']
                    )
    
    # used_time = 0.0
    # used_time_summary = {}
    np_gflops = []
    torch_gflops = []
    blas_gflops = []
    for M, N, K in bench_shapes:
        np.random.seed(0)
        np_A = np.random.rand(M, K).astype(np.float32)
        np_B = np.random.rand(K, N).astype(np.float32)
        np_C = np.zeros((M, N), dtype=np.float32)
        A = torch.tensor(np_A)
        B = torch.tensor(np_B)
        BT = B.transpose(0, 1).contiguous()
        C = torch.tensor(np_C)

        used_time = avg_time_function(np_matmul, np_A, np_B)
        FLOPs = 2 * M * N * K
        GFLOPS = FLOPs / used_time / 1e9
        np_gflops.append(GFLOPS)
        print(f"np.matmul MNK:{M}*{N}*{K}, FLOPs:{FLOPs}, used_time:{used_time:.5f}s, GFLOPS: {GFLOPS:.5f}")

        used_time = avg_time_function(torch_matmul, A, B)
        FLOPs = 2 * M * N * K
        GFLOPS = FLOPs / used_time / 1e9
        torch_gflops.append(GFLOPS)
        print(f"pytorch.matmul MNK:{M}*{N}*{K}, FLOPs:{FLOPs}, used_time:{used_time:.5f}s, GFLOPS: {GFLOPS:.5f}")
    
        used_time = manual_time_function(manual_matmul.native_mnk_matmul, C, A, BT)
        FLOPs = 2 * M * N * K
        GFLOPS = FLOPs / used_time / 1e9
        torch_gflops.append(GFLOPS)
        print(f"native_mnk MNK:{M}*{N}*{K}, FLOPs:{FLOPs}, used_time:{used_time:.5f}s, GFLOPS: {GFLOPS:.5f}")

        used_time = manual_time_function(manual_matmul.native_kmn_matmul, C, A, BT)
        FLOPs = 2 * M * N * K
        GFLOPS = FLOPs / used_time / 1e9
        torch_gflops.append(GFLOPS)
        print(f"native_kmn MNK:{M}*{N}*{K}, FLOPs:{FLOPs}, used_time:{used_time:.5f}s, GFLOPS: {GFLOPS:.5f}")

        used_time = avg_manual_time_function(manual_matmul.blas_matmul, C, A, BT)
        FLOPs = 2 * M * N * K
        GFLOPS = FLOPs / used_time / 1e9
        blas_gflops.append(GFLOPS)
        print(f"blas_matmul MNK:{M}*{N}*{K}, FLOPs:{FLOPs}, used_time:{used_time:.5f}s, GFLOPS: {GFLOPS:.5f}")

        

    # used_time = time_function_v2(manual_matmul.manual_matmul, C, A, BT)
    # print(f"manual_matmul.manual_matmul time: {used_time:.5f}s")

    # used_time = time_function_v2(manual_matmul.manual_matmul_v1, C, A, BT)
    # print('=== profiling manual matmul === ')
    # with torch.autograd.profiler.profile(use_cpu=True, with_flops=True) as prof:
    #     manual_matmul.manual_matmul(C, A, BT)
    # print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=10))

    # print('=== profiling torch matmul === ')
    # with torch.autograd.profiler.profile(use_cpu=True) as prof:
    #     torch_matmul(A, B)
    # print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=10))

    # print(f"manual_matmul.manual_matmul_v1 time: {used_time:.5f}s")
    # print(C)

    # print(torch.matmul(A, B))

    # print(openblas_nt_matmul(A, BT))

if __name__ == "__main__":
    benchmark_matmul()