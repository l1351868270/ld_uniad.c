
import torch
import triton
import triton.language as tl
from triton.runtime import driver
import pdb

device = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(device)
print(properties)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}

def naive_softmax(x: torch.Tensor)->torch.Tensor:
    x_max = x.max(dim=1)[0]
    z = x - x_max
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1)
    ret = numerator / denominator
    return ret

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride,
                   output_row_stride, n_rows, n_cols, 
                   BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
    row_start = tl.program_id(0)
    row_start_ptr = input_ptr + row_start * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_start * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


def softmax(x: torch.Tensor)->torch.Tensor:
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8
    num_stages = 4 if SIZE_SMEM > 200000 else 2
    y = torch.empty_like(x)

    num_programs = n_rows
    softmax_kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE, num_stages=num_stages)
    return y

@triton.jit
def softmax_kernel_v2(output_ptr, input_ptr, input_row_stride,
                   output_row_stride, n_rows, n_cols, 
                   BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=1):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

def softmax_v2(x: torch.Tensor)->torch.Tensor:
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8
    num_stages = 4 if SIZE_SMEM > 200000 else 2
    y = torch.empty_like(x)

    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                       num_stages=num_stages, num_warps=num_warps, grid=(1, ))
        kernel._init_handles()
        n_regs = kernel.n_regs
        size_smem = kernel.metadata.shared
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        # print(f"NUM_REGS:{NUM_REGS}, BLOCK_SIZE={BLOCK_SIZE}, n_regs={n_regs}, size_smem={size_smem}, occupancy={occupancy}")
    occupancy = min(occupancy, SIZE_SMEM // size_smem)
    num_programs = NUM_SM * occupancy
    
    num_programs = min(num_programs, n_rows)
    # print(f"NUM_REGS:{NUM_REGS}, BLOCK_SIZE={BLOCK_SIZE}, n_regs={n_regs}, SIZE_SMEM:{SIZE_SMEM}, size_smem={size_smem}, occupancy={occupancy}, num_programs:{num_programs}")
    softmax_kernel_v2[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE, num_stages=num_stages)
    return y

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[128 * i for i in range(2, 100)],
        line_arg='provider',
        line_vals=['triton', 'triton_v2', 'torch'],
        line_names=['Triton', 'TritonV2', 'Torch'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],
        ylabel='TFLOPS',
        plot_name='softmax-performance',
        args={'M': 4096},
    )
)
def benchmark(N, M, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x))
    if provider == 'triton_v2':
        ms = triton.testing.do_bench(lambda: softmax_v2(x))
    tflops = lambda ms: 5 * x.numel() / ms * 1e-9
    return tflops(ms)

if __name__ == '__main__':
    torch.manual_seed(0)
    M = 1024
    N = 256
    x = torch.randn(M, N, device='cuda')
    y_triton = softmax(x)
    y_triton_v2 = softmax_v2(x)
    y_torch = torch.softmax(x, axis=1)
    # print(y_triton)
    # print(y_torch)
    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
    assert torch.allclose(y_triton_v2, y_torch), (y_triton_v2, y_torch)
    benchmark.run(show_plots=True, print_data=True)

