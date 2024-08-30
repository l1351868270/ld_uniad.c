
import sys
import os
import numpy as np
import torch
import time

from torch.utils.cpp_extension import load

def avg_time_function_help(func, *args):
    start = time.time() * 1000
    func(*args)
    end = time.time() * 1000
    return end - start


def avg_time_function(func, *args, repeat=10):
    # Warmup
    for _ in range(1):
        func(*args)

    used_time = 0.0
    for _ in range(repeat):
        used_time += avg_time_function_help(func, *args)
    return used_time / repeat

def manual_avg_time_function_help(func, *args):
    used_time = func(*args)
    return used_time

def manual_avg_time_function(func, *args, repeat=10):
    # Warmup
    for _ in range(1):
        func(*args)

    used_time = 0.0
    for _ in range(repeat):
        used_time += manual_avg_time_function_help(func, *args)
    return used_time / repeat

def bench_conv2d():
    build_directory = "./build/"
    if not os.path.exists(build_directory):
        os.makedirs(build_directory)
    manual_conv2d = load(name='manual_conv2d', 
                        sources=['./conv2d_api.cpp'], 
                        build_directory=build_directory,
                        verbose=True,
                        extra_include_paths=[],
                        extra_cflags=['-fopenmp', '-Ofast', '-lm', '-std=c++20'],
                        with_cuda=False,
                    )
    
    # N = 1;C = 64;H = 56;W = 56;K = 64;R = 3;S = 3;pad_h = 1;pad_w = 1;U = 1;V = 1;dilation_h = 1;dilation_w = 1

    N = 1;C = 64;H = 56;W = 56;K = 64;R = 3;S = 3;pad_h = 1;pad_w = 1;U = 1;V = 1;dilation_h = 1;dilation_w = 1

    P = (H + 2 * pad_h - dilation_h * (R - 1) - 1) // U + 1
    Q = (W + 2 * pad_w - dilation_w * (S - 1) - 1) // V + 1
    
    v_M = N * P * Q
    v_N = K
    v_K = C * R * S

    gflops = (2.0*v_M*v_N*v_K) * 1e-9
    bytes = 2 * (N * C * H * W + K * C * R * S + N * K * P * Q)
    arithmetic_intensity = 2.0*v_M*v_N*v_K / bytes
    used_time = 0.0
    repeat = 10

    torch.random.manual_seed(0)
    x = torch.randn(N, C, H, W).cpu().float()
    w = torch.randn(K, C, R, S).cpu().float()
    y = torch.zeros(N, K, P, Q).cpu().float()

    w = w.to(memory_format=torch.channels_last)
    x = x.to(memory_format=torch.channels_last)
    y = y.to(memory_format=torch.channels_last)
    # print(f"x: {x.shape}, {x.stride()}, {x}")
    # print(f"w: {w.shape}, {w.stride()}, {w}")
    conv2d = torch.nn.Conv2d(C, K, (R, S), stride=(U, V), padding=(pad_h, pad_w), dilation=(dilation_h, dilation_w), bias=False).cpu().float().to(memory_format=torch.channels_last)
    conv2d.weight.data = w
    y1 = conv2d(x)
    # print(f"torch.conv2d: {y1.shape}, {y1.stride()}, {y1}")
    used_time = avg_time_function(conv2d, x, repeat=repeat)
    print(f"torch.conv2d {N}x{C}x{H}x{W} {K}x{C}x{R}x{S} {N}x{K}x{P}x{Q}, arithmetic_intensity:{arithmetic_intensity:.3f}, im2col MNK: {v_M}x{v_N}x{v_K} GFLOPs:{gflops:.3f}, used_time:{used_time:.3f}ms, TFLOPS:{gflops/used_time:.3f}")

    if torch.has_mkl:
        y_mkl = torch.mkldnn_convolution(x, w, None, (pad_h, pad_w), (U, V), (dilation_h, dilation_w), 1)
        if torch.allclose(y1, y_mkl, atol=1e-2, rtol=1e-2):
            print("Torch and mkldnn_convolution match")
        else:
            print("Torch and mkldnn_convolution differ")
        used_time = avg_time_function(torch.mkldnn_convolution, x, w, None, (pad_h, pad_w), (U, V), (dilation_h, dilation_w), 1, repeat=repeat)
        print(f"torch.mkldnn_convolution {N}x{C}x{H}x{W} {K}x{C}x{R}x{S} {N}x{K}x{P}x{Q}, arithmetic_intensity:{arithmetic_intensity:.3f}, im2col MNK: {v_M}x{v_N}x{v_K} GFLOPs:{gflops:.3f}, used_time:{used_time:.3f}ms, TFLOPS:{gflops/used_time:.3f}")


    manual_conv2d.implicit_gemm_conved(y, x, w, pad_h, pad_w, U, V, dilation_h, dilation_w)
    # print(f"implicit_gemm_conved: {y.shape}, {y.stride()}, {y}")
    if torch.allclose(y1, y, atol=1e-2, rtol=1e-2):
        print("Torch and implicit_gemm_conved match")
    else:
        print("Torch and implicit_gemm_conved differ")

    used_time = manual_avg_time_function(manual_conv2d.implicit_gemm_conved, y, x, w, pad_h, pad_w, U, V, dilation_h, dilation_w, repeat=repeat)
    print(f"implicit_gemm_conved {N}x{C}x{H}x{W} {K}x{C}x{R}x{S} {N}x{K}x{P}x{Q}, arithmetic_intensity:{arithmetic_intensity:.3f}, im2col MNK: {v_M}x{v_N}x{v_K} GFLOPs:{gflops:.3f}, used_time:{used_time:.3f}ms, TFLOPS:{gflops/used_time:.3f}")


if __name__ == "__main__":
    torch.backends.mkldnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = True
    current_device = torch.cpu.current_device()
    print(current_device)

    bench_conv2d()