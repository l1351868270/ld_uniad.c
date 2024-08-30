import torch
import triton
import triton.language as tl

def get_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]


@triton.autotune(
    configs=get_autotune_config(),
    key=['N', 'C', 'H', 'W', 'K', 'P', 'Q', 'R', 'S', 'U', 'V', 'pad_h', 'pad_w', 'dila_h', 'dila_w']
)
@triton.jit
def conv2d_kernel(x_ptr, w_ptr, y_ptr, N, C, H, W, K, P, Q, R, S, U, V, pad_h, pad_w, dila_h, dila_w,
                  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    pid = tl.program_id(0)
    GEMM_M = N * P * Q
    GEMM_K = R * S * C
    GEMM_N = K
    num_pid_n = tl.cdiv(GEMM_N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    gemm_i = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % GEMM_M
    gemm_j = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % GEMM_N

    n = gemm_i // (P * Q)
    npq_residual = gemm_i % (P * Q)
    p = npq_residual // Q
    q = npq_residual % Q
    k = gemm_j

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # tl.device_print(f"accumulator: {accumulator}")
    for idx_k in range(0, tl.cdiv(GEMM_K, BLOCK_SIZE_K)):
        gemm_k = (idx_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)) # % GEMM_K
        r = gemm_k // (S * C)
        rsc_residual = gemm_k % (S * C)
        s = rsc_residual // C
        c = rsc_residual % C
        h = p[:, None] * U + r[None, :] * dila_h - pad_h
        w = q[:, None] * V + s[None, :] * dila_w - pad_w
        mask_x = (h >= 0) & (h < H) & (w >= 0) & (w < W)
        mask_w = (r < R) & (s < S) & (c < C)
        offs_x = n[:, None] * H * W * C + h * W * C + w * C + c
        offs_w = k[None, :] * R * S * C + r[:, None] * S * C + s[:, None] * C + c[:, None]

        x_ptrs = x_ptr + offs_x
        w_ptrs = w_ptr + offs_w

        x_data = tl.load(x_ptrs, mask=mask_x, other=0.0)
        w_data = tl.load(w_ptrs, mask=mask_w[:, None], other=0.0)
        accumulator = tl.dot(x_data, w_data, accumulator)
    c_data = accumulator.to(tl.float16)

    offs_y = gemm_i[:, None] * GEMM_N + gemm_j[None, :]
    mask_y = (gemm_i[:, None] < GEMM_M) & (gemm_j[None, :] < GEMM_N)
    y_ptrs = y_ptr + offs_y
    tl.store(y_ptrs, c_data, mask=mask_y)

def triton_implicit_gemm(x: torch.Tensor, w: torch.Tensor, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    N, C, H, W = x.shape
    K, C, R, S = w.shape
    U, V = stride
    pad_h, pad_w = padding
    dila_h, dila_w = dilation
    P = (H + 2 * pad_h - dila_h * (R - 1) - 1) // U + 1
    Q = (W + 2 * pad_w - dila_w * (S - 1) - 1) // V + 1
    y = torch.empty((N, K, P, Q), device=x.device, dtype=torch.float16).to(memory_format=torch.channels_last)
    v_M = N * P * Q
    v_N = K
    v_K = C * R * S
    grid = lambda META: (triton.cdiv(v_M, META['BLOCK_SIZE_M']) * triton.cdiv(v_N, META['BLOCK_SIZE_N']), )
    conv2d_kernel[grid](x, w, y, N, C, H, W, K, P, Q, R, S, U, V, pad_h, pad_w, dila_h, dila_w)
    return y


ref_lib = 'cuBLAS'
configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=['x_name', 'N', 'C', 'H', 'W', 'K', 'R', 'S', 'U', 'V', 'pad_h', 'pad_w', 'dila_h', 'dila_w'],
        x_vals=[('conv1', 1, 3, 224, 224, 64, 7, 7, 2, 2, 3, 3, 1, 1), 
            ('layer1.0.conv1', 1, 64, 56, 56, 64, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer1.0.conv2', 1, 64, 56, 56, 64, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer1.0.conv3', 1, 64, 56, 56, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer1.0.downsample', 1, 64, 56, 56, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer1.1.conv1', 1, 256, 56, 56, 64, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer1.1.conv2', 1, 64, 56, 56, 64, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer1.1.conv3', 1, 64, 56, 56, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer1.2.conv1', 1, 256, 56, 56, 64, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer1.2.conv2', 1, 64, 56, 56, 64, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer1.2.conv3', 1, 64, 56, 56, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer2.0.conv1', 1, 256, 56, 56, 128, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer2.0.conv2', 1, 128, 28, 28, 128, 3, 3, 2, 2, 1, 1, 1, 1),
            ('layer2.0.conv3', 1, 128, 28, 28, 512, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer2.0.downsample', 1, 256, 28, 28, 512, 1, 1, 2, 2, 0, 0, 1, 1),
            ('layer2.1.conv1', 1, 512, 28, 28, 128, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer2.1.conv2', 1, 128, 28, 28, 128, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer2.1.conv3', 1, 128, 28, 28, 512, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer2.2.conv1', 1, 512, 28, 28, 128, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer2.2.conv2', 1, 128, 28, 28, 128, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer2.2.conv3', 1, 128, 28, 28, 512, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer2.3.conv1', 1, 512, 28, 28, 128, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer2.3.conv2', 1, 128, 28, 28, 128, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer2.3.conv3', 1, 128, 28, 28, 512, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.0.conv1', 1, 512, 28, 28, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.0.conv2', 1, 256, 14, 14, 256, 3, 3, 2, 2, 1, 1, 1, 1),
            ('layer3.0.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.0.downsample', 1, 512, 14, 14, 1024, 1, 1, 2, 2, 0, 0, 1, 1),
            ('layer3.1.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.1.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.1.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.2.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.2.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.2.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.3.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.3.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.3.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.4.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.4.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.4.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.5.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.5.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.5.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.6.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.6.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.6.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.7.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.7.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.7.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.8.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.8.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.8.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.9.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.9.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.9.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.10.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.10.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.10.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.11.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.11.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.11.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.12.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.12.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.12.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.13.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.13.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.13.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.14.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.14.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.14.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.15.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.15.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.15.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.15.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.15.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.15.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.16.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.16.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.16.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.17.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.17.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.17.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.18.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.18.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.18.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.19.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.19.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.19.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.20.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.20.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.20.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.21.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.21.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.21.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.22.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.22.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.22.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer4.0.conv1', 1, 1024, 14, 14, 512, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer4.0.conv2', 1, 512, 7, 7, 512, 3, 3, 2, 2, 1, 1, 1, 1),
            ('layer4.0.conv3', 1, 512, 7, 7, 2048, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer4.0.downsample', 1, 1024, 7, 7, 2048, 1, 1, 2, 2, 0, 0, 1, 1),
            ('layer4.1.conv1', 1, 2048, 7, 7, 512, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer4.1.conv2', 1, 512, 7, 7, 512, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer4.1.conv3', 1, 512, 7, 7, 2048, 1, 1, 1, 1, 0, 0, 1, 1),
            ],
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
        line_vals=[ref_lib.lower(), "triton", 'torch_gemm'],  # Label name for the lines
        line_names=[ref_lib, "Triton", 'Torch_gemm'],  # Line styles
        styles=[("green", "-"), ("blue", "-"), ("red", "-")],  # Line color and style
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance-fp16",
        args={},
    ))
@triton.testing.perf_report(configs)
def benchmark(x_name, N, C, H, W, K, R, S, U, V, pad_h, pad_w, dila_h, dila_w, provider):
    P = (H + 2 * pad_h - dila_h * (R - 1) - 1) // U + 1
    Q = (W + 2 * pad_w - dila_w * (S - 1) - 1) // V + 1
    v_M = N * P * Q
    v_N = K
    v_K = C * R * S
    gflops = (2.0*v_M*v_N*v_K) * 1e-9

    x = torch.randn(N, C, H, W).cuda().half()
    w = torch.randn(K, C, R, S).cuda().half()
    conv2d = torch.nn.Conv2d(C, K, (R, S), stride=(U, V), padding=(pad_h, pad_w), dilation=(dilation_h, dilation_w), bias=False).cuda().half()
    conv2d.weight.data = w
    conv2d = conv2d.to(memory_format=torch.channels_last)
    x = x.to(memory_format=torch.channels_last)
    w = w.to(memory_format=torch.channels_last)

    if provider == ref_lib.lower():
        ms = triton.testing.do_bench(lambda: conv2d(x))

    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: triton_implicit_gemm(x, w, stride=(U, V), padding=(pad_h, pad_w), dilation=(dilation_h, dilation_w)))

    if provider == 'torch_gemm':
        gemm_a = torch.randn(v_M, v_K).cuda().half()
        gemm_b = torch.randn(v_N, v_K).cuda().half().T
        ms = triton.testing.do_bench(lambda: torch.matmul(gemm_a, gemm_b)) 

    perf = lambda ms: gflops / ms
    return perf(ms)

if __name__ == '__main__':
    torch.manual_seed(0)
    N = 1; C = 64; H = 56; W = 56; K = 64; R = 1; S = 1; pad_h = 0; pad_w = 0; U = 1; V = 1; dilation_h = 1; dilation_w = 1
    # N = 1; C = 1; H = 3; W = 3; K = 1; R = 3; S = 3; pad_h = 0; pad_w = 0; U = 1; V = 1; dilation_h = 1; dilation_w = 1

    x = torch.randn(N, C, H, W).cuda().half()
    w = torch.randn(K, C, R, S).cuda().half()
    conv2d = torch.nn.Conv2d(C, K, (R, S), stride=(U, V), padding=(pad_h, pad_w), dilation=(dilation_h, dilation_w), bias=False).cuda().half()
    conv2d.weight.data = w
    y1 = conv2d(x)

    w = w.to(memory_format=torch.channels_last)
    x = x.to(memory_format=torch.channels_last)
    conv2d = conv2d.to(memory_format=torch.channels_last)
    conv2d.weight.data = w
    y2 = conv2d(x)

    y3 = triton_implicit_gemm(x, w, stride=(U, V), padding=(pad_h, pad_w), dilation=(dilation_h, dilation_w))
    
    if torch.allclose(y1, y3, atol=1e-2, rtol=1e-2):
        print("Torch and triton_implicit_gemm match")
    else:
        print("Torch and triton_implicit_gemm differ")
        print(f'torch: shape:{y1.shape}, stride:{y1.stride()}, {y1}')
        print(f'triton_implicit_gemm: shape:{y3.shape}, stride:{y3.stride()}, {y3}')
    
    # benchmark.run(print_data=True, show_plots=False)
    benchmark.run(show_plots=False, print_data=True)
