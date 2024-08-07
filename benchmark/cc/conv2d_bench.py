
import time
import torch
from torch import load
import os

def time_function(func, conv2d, input):
    # Warmup
    for _ in range(2):
        func(conv2d, input)

    start = time.time()
    func(conv2d, input)
    end = time.time()
    return end - start

def torch_conv2d(conv2d, input):
    return conv2d(input)


class NaiveConv2d(object):
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

def benchmark_conv2d():
    N = 6
    C = 3
    H = 928
    W = 1600

    C_out = 64
    ksize = 7
    stride = 2
    padding = 3
    dialation = 1

    conv2d = torch.nn.Conv2d(C, C_out, ksize, stride=stride, padding=padding)
    conv2d.weight = torch.nn.Parameter(torch.rand(C_out, C, ksize, ksize))
    input = torch.rand(N, C, H, W)
    for i in range(10):
        used_time = time_function(torch_conv2d, conv2d, input)
        print(f"torch_conv2d time: {used_time:.5f}")



if __name__ == '__main__':
    benchmark_conv2d()