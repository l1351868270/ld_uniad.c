/*
ncu --csv --log-file build/bench.csv --cache-control=all --clock-control=base --metrics gpu__time_duration.sum ./build/bench_gemm 4096 4096 4096
ncu --csv --log-file build/bench_bank_conflicts.csv --metrics  l1tex__data_bank_conflicts_pipe_lsu_mem_shared,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum ./build/bench_gemm 4096 4096 4096

ncu -f --set full -o build/bench_gemm_report ./build/bench_gemm 4096 4096 4096
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <assert.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "cutlass_gemm_v2.h"
#include "blas.h"


float frand() {
    return (float)rand() / (float)RAND_MAX;
}

void generate_tensor(float * tensor, int M, int N) {
    for (int i = 0; i < M * N; i++) {
        tensor[i] = frand();
    }
}

void generate_range_tensor(float * tensor, int M, int N) {
    for (int i = 0; i < M * N; i++) {
        tensor[i] = i;
    }
}

void print_tensor(half * tensor, int M, int N) {
    printf("[");
    for (int i = 0; i < M; i++) {
        printf("[");
        for (int j = 0; j < N; j++) {
            int offset = i * N + j;
            printf("%.5f, ", __half2float(tensor[offset]));

        }
        printf("],\n");
    }
    printf("]\n");
}

bool check_value(float abs_tol, float rel_tol, half *h_d_c, half *h_c, int m, int n) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float gpu_value = (float)h_d_c[i * n + j];
            float cpu_value = (float)h_c[i * n + j];
            float diff = abs(gpu_value - cpu_value);
            if (diff > max(abs_tol, cpu_value * rel_tol)) {
                std::cout << "blas[" << i << ", " << j << "] = " << gpu_value 
                << ", manual[" << i << ", " << j << "] = " << cpu_value
                << " Abs Diff: " << diff << std::endl;
                return false;
            }
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    srand(0);
    
    int M = 2227200;
    if (argc >= 2) {
        sscanf(argv[1], "%d", &M);
    }
    int N = 128;
    if (argc >= 3) {
        sscanf(argv[2], "%d", &N);
    }
    int K = 128 * 3;
    if (argc >= 4) {
        sscanf(argv[3], "%d", &K);
    }

    std::cout << "M: " << M << ", N: " << N << ", K: " << K << std::endl;

    thrust::host_vector<half> h_A(M * K);
    thrust::host_vector<half> h_B(N * K);
    thrust::host_vector<half> h_C(M * N);
    thrust::host_vector<half> h_C1(M * N);

    for (int j = 0; j < M * K; ++j) h_A[j] = static_cast<half>( 2*(rand() / double(RAND_MAX)) - 1 );
    for (int j = 0; j < N * K; ++j) h_B[j] = static_cast<half>( 2*(rand() / double(RAND_MAX)) - 1 );
    for (int j = 0; j < M * N; ++j) h_C[j] = static_cast<half>(-1);
    for (int j = 0; j < M * N; ++j) h_C1[j] = static_cast<half>(-1);

    thrust::device_vector<half> d_A = h_A;
    thrust::device_vector<half> d_B = h_B;
    thrust::device_vector<half> d_C = h_C;
    double gflops = (2.0*M*N*K) * 1e-9;

    constexpr float abs_tol = 1.0e-0f;
    constexpr float rel_tol = 1.0e-0f;

    thrust::fill(d_C.begin(), d_C.end(), 0.0);
    cublasHandle_t handle;
    cublasCreate(&handle);
    blas_matmul<half>(&handle, d_C.data().get(), d_A.data().get(), d_B.data().get(), M, N, K);
    thrust::copy(d_C.begin(), d_C.end(), h_C.begin());
    // print_tensor(h_C.data(), M, N)
    cublasDestroy(handle);

    thrust::fill(d_C.begin(), d_C.end(), 0.0);
    bench::cutlass_gemm_v2::cutlass_gemm<half>(d_C.data().get(), d_A.data().get(), d_B.data().get(), M, N, K);
    thrust::copy(d_C.begin(), d_C.end(), h_C1.begin());
    // print_tensor(h_C1.data(), M, N);

    if (check_value(abs_tol, rel_tol, h_C.data(), h_C1.data(), M, N)) {
        std::cout << "Test PASSED" << std::endl;
    } else {
        std::cout << "Test FAILED" << std::endl;
    }
    return 0;
}