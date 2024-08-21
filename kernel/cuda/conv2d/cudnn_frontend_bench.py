# ncu -f --set full -o build/cudnn_frontend_report python cudnn_frontend.py
import cudnn
import torch
# print(cudnn.backend_version())

def avg_time_function_help(func, A, B, C):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    func(A, B, C)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def avg_time_function(func, A, B, C, repeat=10):
    # Warmup
    for _ in range(1):
        func(A, B, C)

    used_time = 0.0
    for _ in range(repeat):
        used_time += avg_time_function_help(func, A, B, C)
    return used_time / repeat

def bench_conv2d():
    torch.random.manual_seed(0)
    handle = cudnn.create_handle()

    graph = cudnn.pygraph(
        handle=handle,
        name="cudnn_graph_0",
        io_data_type=cudnn.data_type.HALF,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    frontend_N = 4
    frontend_C = 256
    frontend_H = 232
    frontend_W = 400
    frontend_K = 256
    frontend_R = 3
    frontend_S = 3
    frontend_pad_h = 1
    frontend_pad_w = 1
    frontend_U = 1
    frontend_V = 1
    frontend_dilation_h = 1
    frontend_dilation_w = 1

    frontend_P = (frontend_H + 2 * frontend_pad_h - frontend_dilation_h * (frontend_R - 1) - 1) // frontend_U + 1
    frontend_Q = (frontend_W + 2 * frontend_pad_w - frontend_dilation_w * (frontend_S - 1) - 1) // frontend_V + 1
    
    v_M = frontend_N * frontend_P * frontend_Q
    v_N = frontend_K
    v_K = frontend_C * frontend_R * frontend_S

    gflops = (2.0*v_M*v_N*v_K) * 1e-9
    bytes = 2 * (frontend_N * frontend_C * frontend_H * frontend_W + frontend_K * frontend_C * frontend_R * frontend_S + frontend_N * frontend_K * frontend_P * frontend_Q)
    arithmetic_intensity = 2.0*v_M*v_N*v_K / bytes
    used_time = 0.0
    repeat = 10

    X = graph.tensor(
        name="X",
        dim=[frontend_N, frontend_C, frontend_H, frontend_W],
        stride=[frontend_H * frontend_W * frontend_C, 1, frontend_W * frontend_C, frontend_C],
        data_type=cudnn.data_type.HALF,
    )


    # print(X)

    W = graph.tensor(name="W", dim=[frontend_K, frontend_C, frontend_R, frontend_S], stride=[frontend_R * frontend_S * frontend_C, 1, frontend_S * frontend_C, frontend_C])

    Y = graph.conv_fprop(
        X,
        W,
        padding=[frontend_pad_h, frontend_pad_w],
        stride=[frontend_U, frontend_V],
        dilation=[frontend_dilation_w, frontend_dilation_h],
        compute_data_type=cudnn.data_type.HALF,
    )

    Y.set_output(True)
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    graph.check_support()
    graph.build_plans()


    # print(graph)


    frontend_P = (frontend_H + 2 * frontend_pad_h - frontend_dilation_h * (frontend_R - 1) - 1) // frontend_U + 1
    frontend_Q = (frontend_W + 2 * frontend_pad_w - frontend_dilation_w * (frontend_S - 1) - 1) // frontend_V + 1

    X_gpu = torch.randn(
        frontend_N, frontend_C, frontend_H, frontend_W, requires_grad=False, device="cuda", dtype=torch.float16
    ).to(memory_format=torch.channels_last)
    W_gpu = torch.randn(
        frontend_K, frontend_C, frontend_R, frontend_S, requires_grad=False, device="cuda", dtype=torch.float16
    ).to(memory_format=torch.channels_last)
    Y_gpu = torch.zeros(
        frontend_N, frontend_K, frontend_P, frontend_Q, requires_grad=False, device="cuda", dtype=torch.float16
    ).to(memory_format=torch.channels_last)
    workspace_size = graph.get_workspace_size()
    print(workspace_size)
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    variant_pack = {X: X_gpu, W: W_gpu, Y: Y_gpu}
    used_time = avg_time_function(graph.execute, variant_pack, workspace, handle, repeat=10)
    print(f"torch.conv2d {frontend_N}x{frontend_C}x{frontend_H}x{frontend_W} {frontend_K}x{frontend_C}x{frontend_R}x{frontend_S} "
          f"{frontend_N}x{frontend_K}x{frontend_P}x{frontend_Q}, arithmetic_intensity:{arithmetic_intensity:.3f}, im2col MNK: {v_M}x{v_N}x{v_K} GFLOPs:{gflops:.3f}, used_time:{used_time:.3f}ms, TFLOPS:{gflops/used_time:.3f}")

    # graph.execute(variant_pack, workspace, handle=handle)
    torch.cuda.synchronize()
    # print(Y_gpu)

if __name__ == "__main__":
    bench_conv2d()
