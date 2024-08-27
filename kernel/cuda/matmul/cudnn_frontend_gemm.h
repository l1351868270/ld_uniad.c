#ifndef __LD_UNIAD_CUDNN_FRONTEND_GEMM_H__
#define __LD_UNIAD_CUDNN_FRONTEND_GEMM_H__

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cudnn_frontend.h>
#include <unordered_map>

namespace bench {
namespace cudnn_frontend_gemm {

template <typename T>
double cudnn_gemm(T * C_ptr, T * A_ptr, T * B_ptr, const int M, const int N, const int K) {
    namespace fe = cudnn_frontend;

    // Make cudnn graph
    fe::graph::Graph graph{};
    const int batch = 1;
    // Create the two non-virtual input tensors A and B.
    // There are read from global memory.
    auto A_attributes = fe::graph::Tensor_attributes()
                            .set_name("A")
                            .set_dim({batch, M, K})
                            .set_stride({batch * K, K, 1})
                            .set_data_type(fe::DataType_t::HALF);
    auto A            = graph.tensor(A_attributes);
    auto B_attributes = fe::graph::Tensor_attributes()
                            .set_name("B")
                            .set_dim({batch, K, N})
                            .set_stride({K * N, 1, K})
                            .set_data_type(fe::DataType_t::HALF);
    auto B = graph.tensor(B_attributes);

    auto matmul_attributes = fe::graph::Matmul_attributes().set_compute_data_type(fe::DataType_t::HALF);
    auto C                 = graph.matmul(A, B, matmul_attributes);
    C->set_output(true).set_data_type(fe::DataType_t::HALF);

    graph.validate().is_good();

    cudnnHandle_t handle;
    cudnnCreate(&handle);

    graph.build_operation_graph(handle).is_good();
    graph.create_execution_plans({fe::HeurMode_t::A}).is_good();

    graph.deselect_engines({"eng4_"});
    graph.check_support(handle).is_good();

    graph.build_plans(handle, fe::BuildPlanPolicy_t::ALL).is_good();

    // Run cudnn graph
    void *workSpace;
    int64_t workSpaceSizeInBytes = graph.get_workspace_size();
    cudaMalloc(&workSpace, workSpaceSizeInBytes);
    std::unordered_map<int64_t, void*> variant_pack = {
        {A->get_uid(), (void*)A_ptr}, {B->get_uid(), (void*)B_ptr}, {C->get_uid(), (void*)C_ptr}};

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    graph.execute(handle, variant_pack, workSpace).is_good();
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time_used = 0.0;
    cudaEventElapsedTime(&time_used, start, end);
    cudaFree(workSpace);
    cudnnDestroy(handle);
    return time_used;
}

} // namespace cudnn_frontend_gemm
} // namespace bench


#endif // __LD_UNIAD_CUDNN_FRONTEND_GEMM_H__