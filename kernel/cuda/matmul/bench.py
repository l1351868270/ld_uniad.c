import time
import os
import numpy as np
import torch
from torch.utils.cpp_extension import load_inline, load
import matplotlib.pyplot as plt

def time_function(func, A, B):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Warmup
    for _ in range(1):
        func(A, B)

    start.record()
    func(A, B)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)

def avg_time_function_help(func, A, B):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    func(A, B)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)

def avg_time_function(func, A, B, repeat=10):
    # Warmup
    for _ in range(1):
        func(A, B)

    used_time = 0.0
    for _ in range(repeat):
        used_time += avg_time_function_help(func, A, B)
    return used_time / repeat

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

    # M = 2227200
    # N = 64
    # K = 147
    M = 2048
    N = 2048
    K = 2048
    bench_shapes = [[M, N, K]]

    build_directory = "./kernel/cuda/matmul/build/"
    if not os.path.exists(build_directory):
        os.makedirs(build_directory)
    manual_matmul = load(name='manual_matmul', 
                         sources=['kernel/cuda/matmul/bench.cu'], 
                         build_directory=build_directory,
                         verbose=False,
                         extra_include_paths=['kernel/cuda/matmul', 'kernel/cuda/cutlass/include',
                                              'kernel/cuda/cutlass/tools/util/include', 
                                              'kernel/cuda/ld_kittens',],
                        #  extra_cuda_cflags=['-O3', '-arch=sm_86', '-lcublas', '-std=c++20',
                        #                     '--expt-extended-lambda',
                        #                     '--expt-relaxed-constexpr',
                        #                     '-Xcompiler=-Wno-psabi',
                        #                     '-Xcompiler=-fno-strict-aliasing',
                        #                     '--use_fast_math',
                        #                     '-forward-unknown-to-host-compiler',
                        #                     '-Xptxas=--verbose',
                        #                     '-Xptxas=--warn-on-spills',
                        #                     '-std=c++20',
                        #                     "-U__CUDA_NO_HALF_OPERATORS__",
                        #                     "-U__CUDA_NO_HALF_CONVERSIONS__",
                        #                     "-U__CUDA_NO_HALF2_OPERATORS__",
                        #                     "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                        #                     ],
                        extra_cuda_cflags=['-O3', '-arch=sm_86',  '-lcublas', '-std=c++20',
                                            "-U__CUDA_NO_HALF_OPERATORS__",
                                            "-U__CUDA_NO_HALF_CONVERSIONS__",
                                            "-U__CUDA_NO_HALF2_OPERATORS__",
                                            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                                            ],
                         with_cuda=True,
                    )
    
    # used_time = 0.0
    # used_time_summary = {}
    np_gflops = []
    torch_gflops = []
    blas_gflops = []
    for M, N, K in bench_shapes:
        np.random.seed(0)
        np_A = np.random.rand(M, K).astype(np.float16)
        np_B = np.random.rand(K, N).astype(np.float16)
        np_C = np.zeros((M, N), dtype=np.float16)
        A = torch.tensor(np_A).cuda()
        B = torch.tensor(np_B).cuda()
        BT = B.transpose(0, 1).contiguous()
        C = torch.tensor(np_C).cuda()

        used_time = avg_time_function(np_matmul, np_A, np_B)
        FLOPs = 2 * M * N * K
        TFLOPS = FLOPs / 1e9  / used_time 
        np_gflops.append(TFLOPS)
        print(f"np.matmul MNK:{M}*{N}*{K}, FLOPs:{FLOPs}, used_time:{used_time:.5f}ms, TFLOPS: {TFLOPS:.5f}")

        # used_time = avg_time_function(torch_matmul, A, B)
        # FLOPs = 2 * M * N * K
        # TFLOPS = FLOPs / 1e9  / used_time 
        # torch_gflops.append(TFLOPS)
        # print(f"pytorch.matmul MNK:{M}*{N}*{K}, FLOPs:{FLOPs}, used_time:{used_time:.5f}ms, TFLOPS: {TFLOPS:.5f}")
        # # print(torch_matmul(A, B))

        # # used_time = manual_time_function(manual_matmul.native_mnk_matmul, C, A, BT)
        # # FLOPs = 2 * M * N * K
        # # TFLOPS = FLOPs / used_time / 1e9
        # # torch_gflops.append(TFLOPS)
        # # print(f"native_mnk MNK:{M}*{N}*{K}, FLOPs:{FLOPs}, used_time:{used_time:.5f}ms, TFLOPS: {TFLOPS:.5f}")

        # # used_time = manual_time_function(manual_matmul.native_kmn_matmul, C, A, BT)
        # # FLOPs = 2 * M * N * K
        # # TFLOPS = FLOPs / used_time / 1e9
        # # torch_gflops.append(TFLOPS)
        # # print(f"native_kmn MNK:{M}*{N}*{K}, FLOPs:{FLOPs}, used_time:{used_time:.5f}ms, TFLOPS: {TFLOPS:.5f}")
        # # C.zero_()
        # # manual_matmul.blas_matmul(C, A, BT)
        # # print(C)
        # used_time = avg_manual_time_function(manual_matmul.blas_matmul, C, A, BT)
        # FLOPs = 2 * M * N * K
        # TFLOPS = FLOPs / 1e9  / used_time 
        # blas_gflops.append(TFLOPS)
        # print(f"blas_matmul MNK:{M}*{N}*{K}, FLOPs:{FLOPs}, used_time:{used_time:.5f}ms, TFLOPS: {TFLOPS:.5f}")
        # print(C)

        # used_time = avg_manual_time_function(manual_matmul.cutlass_gemm, C, A, BT)
        # FLOPs = 2 * M * N * K
        # TFLOPS = FLOPs / 1e9  / used_time 
        # blas_gflops.append(TFLOPS)
        # print(f"cutlass_gemm MNK:{M}*{N}*{K}, FLOPs:{FLOPs}, used_time:{used_time:.5f}ms, TFLOPS: {TFLOPS:.5f}")
        # print(C)

        # used_time = avg_manual_time_function(manual_matmul.cutlass_gemm_v2, C, A, BT)
        # FLOPs = 2 * M * N * K
        # TFLOPS = FLOPs / 1e9  / used_time 
        # blas_gflops.append(TFLOPS)
        # print(f"cutlass_gemm_v2 MNK:{M}*{N}*{K}, FLOPs:{FLOPs}, used_time:{used_time:.5f}ms, TFLOPS: {TFLOPS:.5f}")
        # print(C)

if __name__ == "__main__":
    current_device = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(current_device)
    os.environ['TORCH_CUDA_ARCH_LIST'] = f'{device_properties.major}.{device_properties.minor}'
    print(f"current_device:{current_device}, device_properties: {device_properties}")

    benchmark_matmul()