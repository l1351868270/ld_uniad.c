'''
Adapted from https://github.com/cuda-mode/lectures/blob/main/lecture_001/
'''

import time
import os
import numpy as np
import torch
from torch.utils.cpp_extension import load_inline

def time_function(func, A, B):
    # Warmup
    for _ in range(10):
        func(A, B)

    start = time.time()
    func(A, B)
    end = time.time()
    return end - start

def torch_matmul(A, B):
    return torch.matmul(A, B)

def np_matmul(A, B):
    return np.matmul(A, B)

class OpenblasMatmul(object):
    def __init__(self) -> None:
        cpp_source = """
        #include <cblas.h>

        torch::Tensor openblas_matmul(torch::Tensor A, torch::Tensor B) {
            auto A_data = A.data_ptr<float>();
            auto B_data = B.data_ptr<float>();
            auto M = A.size(0);
            auto N = B.size(1);
            auto K = A.size(1);
            auto C = torch::empty({M, N}, torch::kFloat32);
            auto C_data = C.data_ptr<float>();
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A_data, K, B_data, N, 0.0, C_data, N);
            return C;
        }

        """

        build_directory = "./benchmark/cc/build/openblas_matmul"
        if not os.path.exists(build_directory):
            os.makedirs(build_directory)
        self.openblas_matmul_module = load_inline(
            name="openblas_matmul",
            cpp_sources=[cpp_source],
            functions=["openblas_matmul"],
            verbose=True,
            build_directory=build_directory,
        )

    def __call__(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.openblas_matmul_module.openblas_matmul(A, B)

class OpenblasNTMatmul(object):
    def __init__(self) -> None:
        cpp_source = """
        #include <cblas.h>

        torch::Tensor openblas_nt_matmul(torch::Tensor A, torch::Tensor B) {
            auto A_data = A.data_ptr<float>();
            auto B_data = B.data_ptr<float>();
            auto M = A.size(0);
            auto N = B.size(0);
            auto K = A.size(1);
            auto C = torch::empty({M, N}, torch::kFloat32);
            auto C_data = C.data_ptr<float>();
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0, A_data, K, B_data, K, 0.0, C_data, N);
            return C;
        }

        """

        build_directory = "./benchmark/cc/build/openblas_nt_matmul"
        if not os.path.exists(build_directory):
            os.makedirs(build_directory)
        self.openblas_nt_matmul_module = load_inline(
            name="openblas_nt_matmul",
            cpp_sources=[cpp_source],
            functions=["openblas_nt_matmul"],
            verbose=True,
            build_directory=build_directory,
        )

    def __call__(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.openblas_nt_matmul_module.openblas_nt_matmul(A, B)

# def trace_handler(prof: torch.autograd.profiler.profile):
  
#     print(prof.key_averages().table(
#         row_limit=-1))
#     root_path = "./benchmark/cc/profile"
#     if not os.path.exists(root_path):
#         os.makedirs(root_path)
#     prof.export_chrome_trace(f"{root_path}/test_trace_" + str(prof.step_num) + ".json")

def benchmark_matmul():
    # M = 2227200
    # N = 64
    # K = 147
    
    M = 2048
    N = 2048
    K = 2048

    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    for i in range(1000):
        used_time = time_function(np_matmul, A, B)
        print(f"np.matmul time: {used_time:.5f}")
    A = torch.tensor(A)
    B = torch.tensor(B)
    
    used_time = time_function(torch_matmul, A, B)
    print(f"pytorch.matmul time: {used_time:.5f}")

    openblas_matmul = OpenblasMatmul()
    used_time = time_function(openblas_matmul, A, B)
    print(f"openblas_matmul.matmul time: {used_time:.5f}")

    openblas_nt_matmul = OpenblasNTMatmul()
    used_time = time_function(openblas_nt_matmul, A, B.transpose(0, 1))
    print(f"openblas_nt_matmul.matmul time: {used_time:.5f}")

    print("=============")
    print("Profiling torch_matmul")
    print("=============")

    with torch.autograd.profiler.profile(use_cuda=False) as prof:
        torch_matmul(A, B)
    print(prof.key_averages().table(row_limit=-1))


    print("=============")
    print("Profiling openblas_matmul")
    print("=============")

    with torch.autograd.profiler.profile(use_cuda=False) as prof:
        openblas_matmul(A, B)
    print(prof.key_averages().table(row_limit=-1))

    print("=============")
    print("Profiling openblas_nt_matmul")
    print("=============")


    with torch.autograd.profiler.profile(use_cuda=False) as prof:
        openblas_nt_matmul(A, B)
    print(prof.key_averages().table(row_limit=-1))

    root_path = "./benchmark/cc/profile"
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    with torch.profiler.profile(
                                activities=[
                                    torch.profiler.ProfilerActivity.CPU,
                                ],

                                # In this example with wait=1, warmup=1, active=2, repeat=1,
                                # profiler will skip the first step/iteration,
                                # start warming up on the second, record
                                # the third and the forth iterations,
                                # after which the trace will become available
                                # and on_trace_ready (when set) is called;
                                # the cycle repeats starting with the next step

                                schedule=torch.profiler.schedule(
                                    wait=1,
                                    warmup=1,
                                    active=2,
                                    repeat=1),
                                # on_trace_ready=trace_handler
                                on_trace_ready=torch.profiler.tensorboard_trace_handler(root_path)
                                # used when outputting for tensorboard
    ) as p:
        for iter in range(10):
            torch_matmul(A, B)
            # send a signal to the profiler that the next iteration has started
            p.step()

if __name__ == "__main__":
    benchmark_matmul()