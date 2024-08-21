# ncu -f --set full --call-stack -o build/bench_conv2d_report python cudnn_conv2d_test.py
import sys
import os
import numpy as np
import torch
import torch.backends
import torch.backends.cudnn
from torch.utils.cpp_extension import load


def avg_time_function_help(func, A):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    func(A)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def avg_time_function(func, A, repeat=10):
    # Warmup
    for _ in range(1):
        func(A)

    used_time = 0.0
    for _ in range(repeat):
        used_time += avg_time_function_help(func, A)
    return used_time / repeat


def manual_avg_time_function_help(func, y, x, w, pad_h, pad_w, U, V, dilation_h, dilation_w):
    y.fill_(0.0)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    func(y, x, w, pad_h, pad_w, U, V, dilation_h, dilation_w)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def manual_avg_time_function(func, y, x, w, pad_h, pad_w, U, V, dilation_h, dilation_w, repeat=10):
    # Warmup
    for _ in range(1):
        y.fill_(0.0)
        func(y, x, w, pad_h, pad_w, U, V, dilation_h, dilation_w)

    used_time = 0.0
    for _ in range(repeat):
        used_time += manual_avg_time_function_help(func, y, x, w, pad_h, pad_w, U, V, dilation_h, dilation_w)
    return used_time / repeat


def bench_conv2d():
    build_directory = "./build/"
    if not os.path.exists(build_directory):
        os.makedirs(build_directory)
    conda_packages_include = f'{os.path.dirname(sys.executable)}/../lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/include'
    conda_env_include = f'{os.path.dirname(sys.executable)}/../include'
    manual_conv2d = load(name='manual_conv2d', 
                         sources=['./conv2d.cu'], 
                         build_directory=build_directory,
                         verbose=False,
                         extra_include_paths=[
                             conda_packages_include, conda_env_include
                         ],
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
                        extra_cuda_cflags=['-O3', '-arch=sm_86',  '-lcudnn_engines_precompiled', '-lcudnn', '-std=c++20',
                                            "-U__CUDA_NO_HALF_OPERATORS__",
                                            "-U__CUDA_NO_HALF_CONVERSIONS__",
                                            "-U__CUDA_NO_HALF2_OPERATORS__",
                                            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                                            ],
                         with_cuda=True,
                    )
    
    N = 4
    C = 256
    H = 232
    W = 400

    K = 256
    R = 3
    S = 3

    pad_h = 0
    pad_w = 0
    U = 1
    V = 1
    dilation_h = 1
    dilation_w = 1

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
    x = torch.randn(N, C, H, W).cuda().half()
    w = torch.randn(K, C, R, S).cuda().half()
    y = torch.zeros(N, K, P, Q).cuda().half()

    conv2d = torch.nn.Conv2d(C, K, (R, S), stride=(U, V), padding=(pad_h, pad_w), dilation=(dilation_h, dilation_w), bias=False).cuda().half()
    conv2d.weight.data = w
    y1 = conv2d(x)
    print(f"torch.conv2d: {y1}")
    used_time = avg_time_function(conv2d, x, repeat)
    print(f"torch.conv2d {N}x{C}x{H}x{W} {K}x{C}x{R}x{S} {N}x{K}x{P}x{Q}, arithmetic_intensity:{arithmetic_intensity:.3f}, im2col MNK: {v_M}x{v_N}x{v_K} GFLOPs:{gflops:.3f}, used_time:{used_time:.3f}ms, TFLOPS:{gflops/used_time:.3f}")

    y.fill_(0.0)
    manual_conv2d.cudnn_conv2d_nchw(y, x, w, pad_h, pad_w, U, V, dilation_h, dilation_w)
    print(f"cudnn_conv2d_nchw: {y}")
    used_time = manual_avg_time_function(manual_conv2d.cudnn_conv2d_nchw, y, x, w, pad_h, pad_w, U, V, dilation_h, dilation_w, repeat)
    print(f"cudnn_conv2d_nchw {N}x{C}x{H}x{W} {K}x{C}x{R}x{S} {N}x{K}x{P}x{Q}, arithmetic_intensity:{arithmetic_intensity:.3f}, im2col MNK: {v_M}x{v_N}x{v_K} GFLOPs:{gflops:.3f}, used_time:{used_time:.3f}ms, TFLOPS:{gflops/used_time:.3f}")

    y.fill_(0.0)
    manual_conv2d.cudnn_conv2d_nchw_best(y, x, w, pad_h, pad_w, U, V, dilation_h, dilation_w)
    print(f"cudnn_conv2d_nchw_best: {y}")
    # used_time = manual_avg_time_function(manual_conv2d.cudnn_conv2d_nchw_best, y, x, w, pad_h, pad_w, U, V, dilation_h, dilation_w, repeat)
    # print(f"cudnn_conv2d_nchw_best {N}x{C}x{H}x{W} {K}x{C}x{R}x{S} {N}x{K}x{P}x{Q}, arithmetic_intensity:{arithmetic_intensity:.3f}, im2col MNK: {v_M}x{v_N}x{v_K} GFLOPs:{gflops:.3f}, used_time:{used_time:.3f}ms, TFLOPS:{gflops/used_time:.3f}")

    conv2d = conv2d.to(memory_format=torch.channels_last)
    w = w.to(memory_format=torch.channels_last)
    x = x.to(memory_format=torch.channels_last)
    y = y.to(memory_format=torch.channels_last)
    y2 = conv2d(x)
    print(f"torch.conv2d channels_last: {y2}")
    # used_time = avg_time_function(conv2d, x, repeat)
    # print(f"torch.conv2d channels_last {N}x{C}x{H}x{W} {K}x{C}x{R}x{S} {N}x{K}x{P}x{Q}, arithmetic_intensity:{arithmetic_intensity:.3f}, im2col MNK: {v_M}x{v_N}x{v_K} GFLOPs:{gflops:.3f}, used_time:{used_time:.3f}ms, TFLOPS:{gflops/used_time:.3f}")
    
    

    y.fill_(0.0)
    manual_conv2d.cudnn_conv2d_nhwc(y, x, w, pad_h, pad_w, U, V, dilation_h, dilation_w)
    print(f"cudnn_conv2d_nhwc: {y}")
    # # used_time = manual_avg_time_function(manual_conv2d.cudnn_conv2d_nhwc, y_nhwc, x_nhwc, w_nhwc, pad_h, pad_w, U, V, dilation_h, dilation_w, repeat)
    # # print(f"cudnn_conv2d_nhwc {N}x{C}x{H}x{W} {K}x{C}x{R}x{S} {N}x{K}x{P}x{Q}, arithmetic_intensity:{arithmetic_intensity:.3f}, im2col MNK: {v_M}x{v_N}x{v_K} GFLOPs:{gflops:.3f}, used_time:{used_time:.3f}ms, TFLOPS:{gflops/used_time:.3f}")

    y.fill_(0.0)
    manual_conv2d.cudnn_conv2d_nhwc_best(y, x, w, pad_h, pad_w, U, V, dilation_h, dilation_w)
    print(f"cudnn_conv2d_nhwc_best: {y}")
    # # used_time = manual_avg_time_function(manual_conv2d.cudnn_conv2d_nhwc_best, y_nhwc, x_nhwc, w_nhwc, pad_h, pad_w, U, V, dilation_h, dilation_w, repeat)
    # # print(f"cudnn_conv2d_nhwc_best {N}x{C}x{H}x{W} {K}x{C}x{R}x{S} {N}x{K}x{P}x{Q}, arithmetic_intensity:{arithmetic_intensity:.3f}, im2col MNK: {v_M}x{v_N}x{v_K} GFLOPs:{gflops:.3f}, used_time:{used_time:.3f}ms, TFLOPS:{gflops/used_time:.3f}")

    y.fill_(0.0)
    manual_conv2d.cudnn_frontend_conv2d(y, x, w, pad_h, pad_w, U, V, dilation_h, dilation_w)
    print(f"cudnn_frontend_conv2d: {y}")
    # used_time = manual_avg_time_function(manual_conv2d.cudnn_frontend_conv2d, y_nhwc, x_nhwc, w_nhwc, pad_h, pad_w, U, V, dilation_h, dilation_w, repeat)
    # print(f"cudnn_frontend_conv2d {N}x{C}x{H}x{W} {K}x{C}x{R}x{S} {N}x{K}x{P}x{Q}, arithmetic_intensity:{arithmetic_intensity:.3f}, im2col MNK: {v_M}x{v_N}x{v_K} GFLOPs:{gflops:.3f}, used_time:{used_time:.3f}ms, TFLOPS:{gflops/used_time:.3f}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    current_device = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(current_device)
    os.environ['TORCH_CUDA_ARCH_LIST'] = f'{device_properties.major}.{device_properties.minor}'
    print(f"current_device:{current_device}, device_properties: {device_properties}")

    bench_conv2d()