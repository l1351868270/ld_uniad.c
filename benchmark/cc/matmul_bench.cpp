#include <torch/extension.h>
#include "matmul.h"

void manual_matmul(torch::Tensor C, torch::Tensor A, torch::Tensor B) {
    auto C_data = C.data_ptr<float>();
    auto A_data = A.data_ptr<float>();
    auto B_data = B.data_ptr<float>();
    auto M = C.size(0);
    auto N = C.size(1);
    auto K = A.size(1);
    cc_matmul_fwd(C_data, A_data, B_data, M, N, K);
}

void manual_matmul_v1(torch::Tensor C, torch::Tensor A, torch::Tensor B) {
    auto C_data = C.data_ptr<float>();
    auto A_data = A.data_ptr<float>();
    auto B_data = B.data_ptr<float>();
    auto M = C.size(0);
    auto N = C.size(1);
    auto K = A.size(1);
    cc_matmul_v1_fwd(C_data, A_data, B_data, M, N, K);
}
        
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("manual_matmul", torch::wrap_pybind_function(manual_matmul), "manual_matmul");
m.def("manual_matmul_v1", torch::wrap_pybind_function(manual_matmul_v1), "manual_matmul_v1");
}