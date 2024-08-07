#include <torch/extension.h>
// #include "native_mnk.h"
// #include "native_kmn.h"
#include "blas.h"
#include "cutlass_gemm.h"
#include "cutlass_gemm_v2.h"

// double py_native_mnk_matmul(torch::Tensor C, torch::Tensor A, torch::Tensor B) {
//     auto C_data = C.data_ptr<float>();
//     auto A_data = A.data_ptr<float>();
//     auto B_data = B.data_ptr<float>();
//     auto M = C.size(0);
//     auto N = C.size(1);
//     auto K = A.size(1);
//     return native_mnk_matmul(C_data, A_data, B_data, M, N, K);
// }

// double py_native_kmn_matmul(torch::Tensor C, torch::Tensor A, torch::Tensor B) {
//     auto C_data = C.data_ptr<float>();
//     auto A_data = A.data_ptr<float>();
//     auto B_data = B.data_ptr<float>();
//     auto M = C.size(0);
//     auto N = C.size(1);
//     auto K = A.size(1);
//     return native_kmn_matmul(C_data, A_data, B_data, M, N, K);
// }

double py_blas_matmul(torch::Tensor C, torch::Tensor A, torch::Tensor B) {
    auto C_data = C.data_ptr<float>();
    auto A_data = A.data_ptr<float>();
    auto B_data = B.data_ptr<float>();
    auto M = C.size(0);
    auto N = C.size(1);
    auto K = A.size(1);
    cublasHandle_t handle;
    cublasCreate(&handle);
    return blas_matmul(&handle, C_data, A_data, B_data, M, N, K);
    cublasDestroy(handle);
}

// void manual_matmul_v1(torch::Tensor C, torch::Tensor A, torch::Tensor B) {
//     auto C_data = C.data_ptr<float>();
//     auto A_data = A.data_ptr<float>();
//     auto B_data = B.data_ptr<float>();
//     auto M = C.size(0);
//     auto N = C.size(1);
//     auto K = A.size(1);
//     cc_matmul_v1_fwd(C_data, A_data, B_data, M, N, K);
// }

double py_cutlass_gemm(torch::Tensor C, torch::Tensor A, torch::Tensor B) {
    auto C_data = C.data_ptr<float>();
    auto A_data = A.data_ptr<float>();
    auto B_data = B.data_ptr<float>();
    auto M = C.size(0);
    auto N = C.size(1);
    auto K = A.size(1);
    return cutlass_gemm(C_data, A_data, B_data, M, N, K);
}

double py_cutlass_gemm_v2(torch::Tensor C, torch::Tensor A, torch::Tensor B) {
    auto C_data = C.data_ptr<float>();
    auto A_data = A.data_ptr<float>();
    auto B_data = B.data_ptr<float>();
    auto M = C.size(0);
    auto N = C.size(1);
    auto K = A.size(1);
    return cutlass_gemm_v2(C_data, A_data, B_data, M, N, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
// m.def("native_mnk_matmul", torch::wrap_pybind_function(py_native_mnk_matmul), "native_mnk_matmul");
// m.def("native_kmn_matmul", torch::wrap_pybind_function(py_native_kmn_matmul), "native_kmn_matmul");
m.def("blas_matmul", torch::wrap_pybind_function(py_blas_matmul), "blas_matmul");
m.def("cutlass_gemm", torch::wrap_pybind_function(py_cutlass_gemm), "cutlass_gemm");
m.def("cutlass_gemm_v2", torch::wrap_pybind_function(py_cutlass_gemm_v2), "cutlass_gemm_v2");
}