import time
import os
import numpy as np
import torch
from torch.utils.cpp_extension import load_inline, load

def time_function(func, A, B):
    # Warmup
    for _ in range(10):
        func(A, B)

    start = time.time()
    func(A, B)
    end = time.time()
    return end - start

def time_function_v2(func, C, A, B):
    # Warmup
    # for _ in range(10):
    #     func(C, A, B)

    start = time.time()
    func(C, A, B)
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
            verbose=False,
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
            verbose=False,
            build_directory=build_directory,
        )

    def __call__(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.openblas_nt_matmul_module.openblas_nt_matmul(A, B)


def benchmark_matmul():
    M = 2227200
    N = 64
    K = 147
    
    # M = 2048
    # N = 2048
    # K = 2048
    np.random.seed(0)
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    used_time = time_function(np_matmul, A, B)
    print(f"np.matmul time: {used_time:.5f}s")
    A = torch.tensor(A)
    B = torch.tensor(B)
    BT = B.transpose(0, 1).contiguous()
    C = torch.tensor(C)
    
    used_time = time_function(torch_matmul, A, B)
    print(f"pytorch.matmul time: {used_time:.5f}s")

    openblas_matmul = OpenblasMatmul()
    used_time = time_function(openblas_matmul, A, B)
    print(f"openblas_matmul.matmul time: {used_time:.5f}s")

    openblas_nt_matmul = OpenblasNTMatmul()
    used_time = time_function(openblas_nt_matmul, A, BT)
    print(f"openblas_nt_matmul.matmul time: {used_time:.5f}s")

    build_directory = "./benchmark/cc/build/manual_matmul"
    if not os.path.exists(build_directory):
        os.makedirs(build_directory)
    manual_matmul = load(name='manual_matmul', 
                         sources=['benchmark/cc/matmul_bench.cpp'], 
                         build_directory=build_directory,
                         verbose=False,
                         extra_include_paths=['kernel/cc']
                    )
    # used_time = time_function_v2(manual_matmul.manual_matmul, C, A, BT)
    # print(f"manual_matmul.manual_matmul time: {used_time:.5f}s")

    # used_time = time_function_v2(manual_matmul.manual_matmul_v1, C, A, BT)
    print('=== profiling manual matmul === ')
    with torch.autograd.profiler.profile(use_cpu=True, with_flops=True) as prof:
        manual_matmul.manual_matmul(C, A, BT)
    print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=10))

    print('=== profiling torch matmul === ')
    with torch.autograd.profiler.profile(use_cpu=True) as prof:
        torch_matmul(A, B)
    print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=10))

    # print(f"manual_matmul.manual_matmul_v1 time: {used_time:.5f}s")
    # print(C)

    # print(torch.matmul(A, B))

    # print(openblas_nt_matmul(A, BT))

if __name__ == "__main__":
    benchmark_matmul()