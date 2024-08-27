import math
import os
import torch
import torch_tensorrt

import torch.utils.benchmark as benchmark
from calflops import calculate_flops


def benchmark_forward(
    fn, *inputs, repeats=10, desc="", verbose=True, amp=False, amp_dtype=torch.float16, **kwinputs
):
    """Use Pytorch Benchmark on the forward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Forward pass")

    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)

    t = benchmark.Timer(
        stmt="fn_amp(*inputs, **kwinputs)",
        globals={"fn_amp": amp_wrapper, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m

def benchmark_memory(fn, *inputs, desc="", verbose=True, **kwinputs):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fn(*inputs, **kwinputs)
    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated() / ((2**20) * 1000)
    if verbose:
        print(f"{desc} max memory: {mem}GB")
    torch.cuda.empty_cache()
    return mem

def time_fwd_bwd(func, *args, **kwargs):
    time_f = benchmark_forward(func, *args, **kwargs)
    return time_f[1].mean

def flops(model, input_shape):
    flops, macs, params = calculate_flops(model=model, 
                                      input_shape=input_shape,
                                      output_as_string=False,
                                      output_precision=4,
                                      print_results=False,
                                      print_detailed=False,)
    return flops, macs, params

def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0

if __name__ == '__main__':
    import torchvision.models as models
    # models.vit_b_32
    model = models.resnet101(pretrained=True).half().eval().to("cuda")
    model.eval()
    print(model)
    input_shape = (1, 3, 224, 224)
    dummy_input = torch.randn(*input_shape).to("cuda").half()

    g, _, params = flops(model, input_shape)
    model_bytes = 2 * params / 1024 / 1024 / 1024
    with torch.no_grad():
        benchmark_memory(model, dummy_input, desc="torch ResNet101", verbose=True)
        f = time_fwd_bwd(model, dummy_input, desc="torch ResNet101", verbose=False)
        
        TFLOPS = efficiency(g, f)
        print(f"[torch][ResNet101]: Fwd: {f:.4f} s, FLOPs: {g}, TFLOPS: {TFLOPS:.4f}, model_size:{model_bytes}GB, bindwith:{model_bytes/f:.4f}")
        
    inputs = [dummy_input]
    enabled_precisions = {torch.half}
    debug = False
    workspace_size = 20 << 30
    min_block_size = 7
    torch_executed_ops = {}
    import logging
    logging.basicConfig(level=logging.WARNING)
    torch_tensorrt.logging._LOGGER.setLevel(logging.WARNING)
    optimized_model_path = "build/resnet101_trt.ep"
    new_inputs = torch.randn(input_shape).half().to("cuda")
    if not os.path.exists(optimized_model_path):
        optimized_model = torch_tensorrt.compile(
            model,
            "dynamo",
            inputs=inputs,
            enabled_precisions=enabled_precisions,
            workspace_size=workspace_size,
            min_block_size=min_block_size,
            torch_executed_ops=torch_executed_ops,
        )
        print(optimized_model)
        torch_tensorrt.save(optimized_model, optimized_model_path, inputs=inputs)
    else:
        optimized_model = torch.export.load(optimized_model_path).module()
    
    new_outputs = optimized_model(new_inputs)
    with torch.no_grad():
        benchmark_memory(optimized_model, new_inputs, desc="torch ResNet101", verbose=True)
        f = time_fwd_bwd(optimized_model, new_inputs, desc="torch ResNet101", verbose=False)
        TFLOPS = efficiency(g, f)
        print(f"[torch_tensorrt][ResNet101]: Fwd: {f:.4f} s, FLOPs: {g}, TFLOPS: {TFLOPS:.4f}")