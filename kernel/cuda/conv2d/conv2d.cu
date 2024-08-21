#include <torch/extension.h>
#include "cudnn_conv2d.h"
#include "cudnn_conv2d_nchw.h"
#include "cudnn_conv2d_nchw_best.h"
#include "cudnn_conv2d_nhwc.h"
#include "cudnn_conv2d_nhwc_best.h"
#include "cudnn_frontend_conv2d.h"

double py_cudnn_conv2d(torch::Tensor y, torch::Tensor x, torch::Tensor w, int pad_h, int pad_w, int U, int V, int dilation_h, int dilation_w) {
    c10::Half *y_ptr = y.data_ptr<c10::Half>();
    c10::Half *x_ptr = x.data_ptr<c10::Half>();
    c10::Half *w_ptr = w.data_ptr<c10::Half>();

    half* y_hf = reinterpret_cast<half*>(y_ptr);
    half* x_hf = reinterpret_cast<half*>(x_ptr);
    half* w_hf = reinterpret_cast<half*>(w_ptr);
    // auto y_data = (half*)y.data_ptr();
    // auto x_data = (half*)x.data_ptr();
    // auto w_data = (half*)w.data_ptr();
    auto N = x.size(0);
    auto H = x.size(2);
    auto W = x.size(3);
    auto C = x.size(1);

    auto K = w.size(0);
    auto R = w.size(2);
    auto S = w.size(3);

    // printf("N:%d, H:%d, W:%d, C:%d, K:%d, R:%d, S:%d, pad_h:%d, pad_w:%d, U:%d, V:%d, dilation_h:%d, dilation_w:%d\n", 
    //         N, H, W, C, K, R, S, pad_h, pad_w, U, V, dilation_h, dilation_w);
    return bench::cudnn_conv2d::cudnn_conv2d<half>(y_hf, x_hf, w_hf, N, H, W, C, 
                       K, R, S, pad_h, pad_w, U, V, dilation_h, dilation_w);
}

double py_cudnn_conv2d_nchw(torch::Tensor y, torch::Tensor x, torch::Tensor w, int pad_h, int pad_w, int U, int V, int dilation_h, int dilation_w) {
    c10::Half *y_ptr = y.data_ptr<c10::Half>();
    c10::Half *x_ptr = x.data_ptr<c10::Half>();
    c10::Half *w_ptr = w.data_ptr<c10::Half>();

    half* y_hf = reinterpret_cast<half*>(y_ptr);
    half* x_hf = reinterpret_cast<half*>(x_ptr);
    half* w_hf = reinterpret_cast<half*>(w_ptr);
    // auto y_data = (half*)y.data_ptr();
    // auto x_data = (half*)x.data_ptr();
    // auto w_data = (half*)w.data_ptr();
    auto N = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);
    
    auto K = w.size(0);
    auto R = w.size(2);
    auto S = w.size(3);

    // printf("N:%d, H:%d, W:%d, C:%d, K:%d, R:%d, S:%d, pad_h:%d, pad_w:%d, U:%d, V:%d, dilation_h:%d, dilation_w:%d\n", 
    //         N, H, W, C, K, R, S, pad_h, pad_w, U, V, dilation_h, dilation_w);
    return bench::cudnn_conv2d_nchw::cudnn_conv2d<half>(y_hf, x_hf, w_hf, N, H, W, C, 
                       K, R, S, pad_h, pad_w, U, V, dilation_h, dilation_w);
}

double py_cudnn_conv2d_nchw_best(torch::Tensor y, torch::Tensor x, torch::Tensor w, int pad_h, int pad_w, int U, int V, int dilation_h, int dilation_w) {
    c10::Half *y_ptr = y.data_ptr<c10::Half>();
    c10::Half *x_ptr = x.data_ptr<c10::Half>();
    c10::Half *w_ptr = w.data_ptr<c10::Half>();

    half* y_hf = reinterpret_cast<half*>(y_ptr);
    half* x_hf = reinterpret_cast<half*>(x_ptr);
    half* w_hf = reinterpret_cast<half*>(w_ptr);
    // auto y_data = (half*)y.data_ptr();
    // auto x_data = (half*)x.data_ptr();
    // auto w_data = (half*)w.data_ptr();
    auto N = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);
    
    auto K = w.size(0);
    auto R = w.size(2);
    auto S = w.size(3);

    // printf("N:%d, H:%d, W:%d, C:%d, K:%d, R:%d, S:%d, pad_h:%d, pad_w:%d, U:%d, V:%d, dilation_h:%d, dilation_w:%d\n", 
    //         N, H, W, C, K, R, S, pad_h, pad_w, U, V, dilation_h, dilation_w);
    return bench::cudnn_conv2d_nchw_best::cudnn_conv2d<half>(y_hf, x_hf, w_hf, N, H, W, C, 
                       K, R, S, pad_h, pad_w, U, V, dilation_h, dilation_w);
}

double py_cudnn_conv2d_nhwc(torch::Tensor y, torch::Tensor x, torch::Tensor w, int pad_h, int pad_w, int U, int V, int dilation_h, int dilation_w) {
    c10::Half *y_ptr = y.data_ptr<c10::Half>();
    c10::Half *x_ptr = x.data_ptr<c10::Half>();
    c10::Half *w_ptr = w.data_ptr<c10::Half>();

    half* y_hf = reinterpret_cast<half*>(y_ptr);
    half* x_hf = reinterpret_cast<half*>(x_ptr);
    half* w_hf = reinterpret_cast<half*>(w_ptr);
    // auto y_data = (half*)y.data_ptr();
    // auto x_data = (half*)x.data_ptr();
    // auto w_data = (half*)w.data_ptr();
    auto N = x.size(0);
    
    auto H = x.size(1);
    auto W = x.size(2);
    
    auto K = w.size(0);
    auto R = w.size(1);
    auto S = w.size(2);
    auto C = x.size(3);

    // printf("N:%d, H:%d, W:%d, C:%d, K:%d, R:%d, S:%d, pad_h:%d, pad_w:%d, U:%d, V:%d, dilation_h:%d, dilation_w:%d\n", 
    //         N, H, W, C, K, R, S, pad_h, pad_w, U, V, dilation_h, dilation_w);
    return bench::cudnn_conv2d_nhwc::cudnn_conv2d<half>(y_hf, x_hf, w_hf, N, H, W, C, 
                       K, R, S, pad_h, pad_w, U, V, dilation_h, dilation_w);
    return 0;
}

double py_cudnn_conv2d_nhwc_best(torch::Tensor y, torch::Tensor x, torch::Tensor w, int pad_h, int pad_w, int U, int V, int dilation_h, int dilation_w) {
    c10::Half *y_ptr = y.data_ptr<c10::Half>();
    c10::Half *x_ptr = x.data_ptr<c10::Half>();
    c10::Half *w_ptr = w.data_ptr<c10::Half>();

    half* y_hf = reinterpret_cast<half*>(y_ptr);
    half* x_hf = reinterpret_cast<half*>(x_ptr);
    half* w_hf = reinterpret_cast<half*>(w_ptr);
    // auto y_data = (half*)y.data_ptr();
    // auto x_data = (half*)x.data_ptr();
    // auto w_data = (half*)w.data_ptr();
    auto N = x.size(0);
    
    auto H = x.size(1);
    auto W = x.size(2);
    
    auto K = w.size(0);
    auto R = w.size(1);
    auto S = w.size(2);
    auto C = x.size(3);

    // printf("N:%d, H:%d, W:%d, C:%d, K:%d, R:%d, S:%d, pad_h:%d, pad_w:%d, U:%d, V:%d, dilation_h:%d, dilation_w:%d\n", 
    //         N, H, W, C, K, R, S, pad_h, pad_w, U, V, dilation_h, dilation_w);
    return bench::cudnn_conv2d_nhwc_best::cudnn_conv2d<half>(y_hf, x_hf, w_hf, N, H, W, C, 
                       K, R, S, pad_h, pad_w, U, V, dilation_h, dilation_w);
    return 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("cudnn_conv2d", torch::wrap_pybind_function(py_cudnn_conv2d), "cudnn_conv2d");
m.def("cudnn_conv2d_nchw", torch::wrap_pybind_function(py_cudnn_conv2d_nchw), "cudnn_conv2d_nchw");
m.def("cudnn_conv2d_nchw_best", torch::wrap_pybind_function(py_cudnn_conv2d_nchw_best), "cudnn_conv2d_nchw_best");
m.def("cudnn_conv2d_nhwc", torch::wrap_pybind_function(py_cudnn_conv2d_nhwc), "cudnn_conv2d_nhwc");
m.def("cudnn_conv2d_nhwc_best", torch::wrap_pybind_function(py_cudnn_conv2d_nhwc_best), "cudnn_conv2d_nhwc_best");
}