#ifndef __LD_UNIAD_CUDNN_FRONTEND_CONV2D_H__
#define __LD_UNIAD_CUDNN_FRONTEND_CONV2D_H__

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cudnn_frontend.h>
#include <unordered_map>

namespace bench {
namespace cudnn_frontend_conv2d {

template <typename T>
double cudnn_conv2d(T * y, T * x, const T * w, int N, int H, int W, int C, 
                    int K, int R, int S, int pad_h, int pad_w, int U, int V, int dilation_h, int dilation_w) {
    namespace fe = cudnn_frontend;
    auto build_new_graph = [=](cudnnHandle_t handle) {
        auto graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(fe::DataType_t::HALF).set_compute_data_type(fe::DataType_t::HALF);

        auto X = graph->tensor(fe::graph::Tensor_attributes()
                                    .set_name("image")
                                    .set_dim({N, C, H, W})
                                    .set_stride({H * W * C, 1, W * C, C}));

        auto W = graph->tensor(fe::graph::Tensor_attributes()
                                    .set_name("filter")
                                    .set_dim({K, C, R, S})
                                    .set_stride({R * S * C, 1, S * C, C}));

        auto conv_options =
            fe::graph::Conv_fprop_attributes().set_padding({pad_h, pad_w}).set_stride({U, V}).set_dilation({dilation_h, dilation_w});
        auto Y = graph->conv_fprop(X, W, conv_options);

        Y->set_output(true);

        graph->validate().is_good();

        graph->build_operation_graph(handle).is_good();

        graph->create_execution_plans({fe::HeurMode_t::A}).is_good();

        graph->check_support(handle).is_good();

        graph->build_plans(handle).is_good();

        return std::make_tuple(graph, X, W, Y);
    };
    
    cudnnHandle_t handle;
    cudnnStatus_t err;
    err = cudnnCreate(&handle);
    if (err != CUDNN_STATUS_SUCCESS) {
        printf("cudnnCreate failed: %s\n", cudnnGetErrorString(err));
        exit(1);
    }
    auto [graph, frontend_X, frontend_W, frontend_Y] = build_new_graph(handle);
    std::unordered_map<int64_t, void*> variant_pack = {
        {frontend_X->get_uid(), (void*)x}, {frontend_W->get_uid(), (void*)w}, {frontend_Y->get_uid(), (void*)y}};
    
    void *workSpace;
    int64_t workSpaceSizeInBytes = graph->get_workspace_size();
    // printf("cudnn_frontend_conv2d: %ld %ld %ld %ld\n", frontend_X->get_uid(), frontend_W->get_uid(), frontend_Y->get_uid(), workSpaceSizeInBytes);
    cudaMalloc(&workSpace, workSpaceSizeInBytes);
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    graph->execute(handle, variant_pack, workSpace).is_good();
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time_used = 0.0;
    cudaEventElapsedTime(&time_used, start, end);
    cudaFree(workSpace);
    cudnnDestroy(handle);
    return time_used;
}

} // namespace cudnn_frontend_conv2d
} // namespace bench

#endif // __LD_UNIAD_CUDNN_FRONTEND_CONV2D_H__