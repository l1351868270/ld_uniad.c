#include <torch/extension.h>
#include "implicit_gemm.h"

double py_implicit_gemm_conv2d(torch::Tensor y, torch::Tensor x, torch::Tensor w, int pad_h, int pad_w, int U, int V, int dilation_h, int dilation_w) {
    auto y_data = (float*)y.data_ptr();
    auto x_data = (float*)x.data_ptr();
    auto w_data = (float*)w.data_ptr();

    auto N = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);

    auto K = w.size(0);
    auto R = w.size(2);
    auto S = w.size(3);
    
    auto P = floor((H + 2 * pad_h - dilation_h * (R - 1) - 1) / U + 1);
    auto Q = floor((W + 2 * pad_w - dilation_w * (S - 1) - 1) / V + 1);

    return bench::cc_implicit_gemm::conv2d_fwd(y_data, x_data, w_data, N, C, H, W,
                       K, P, Q, R, S, U, V, pad_h, pad_w, dilation_h, dilation_w, "zeros");
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("implicit_gemm_conved", torch::wrap_pybind_function(py_implicit_gemm_conv2d), "implicit_gemm_conved");
}