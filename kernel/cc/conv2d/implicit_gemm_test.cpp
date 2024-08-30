#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <assert.h>
#include <cmath>
#include "implicit_gemm.h"

void print_tensor(float * tensor, int N, int C, int H, int W) {
    printf("[");
    for (int n = 0; n < N; n++) {
        printf("[");
        for (int h = 0; h < H; h++) {
            printf("[");
            for (int w = 0; w < W; w++) {
                printf("[");
                for (int c = 0; c < C; c++) {
                    int offset = n * H * W * C + h * W * C + w * C + c;
                    printf("%.3f, ", tensor[offset]);
                }
                printf("],\n");
            }
            printf("],\n");
        }
        printf("],\n");
    }
    printf("]\n");
}

int main(int argc, char ** argv) {
    srand(0);

    int N = 1;
    if (argc >= 2) {
        sscanf(argv[1], "%d", &N);
    }
    int H = 56;
    if (argc >= 3) {
        sscanf(argv[2], "%d", &H);
    }
    int W = 56;
    if (argc >= 4) {
        sscanf(argv[3], "%d", &W);
    }
    int C = 64;
    if (argc >= 5) {
        sscanf(argv[4], "%d", &C);
    }
    int K = 64;
    if (argc >= 6) {
        sscanf(argv[5], "%d", &K);
    }
    int R = 3;
    if (argc >= 7) {
        sscanf(argv[6], "%d", &R);
    }
    int S = 3;
    if (argc >= 8) {
        sscanf(argv[7], "%d", &S);
    }
    int pad_h = 1;
    if (argc >= 9) {
        sscanf(argv[8], "%d", &pad_h);
    }
    int pad_w = 1;
    if (argc >= 10) {
        sscanf(argv[9], "%d", &pad_w);
    }
    int U = 1;
    if (argc >= 11) {
        sscanf(argv[10], "%d", &U);
    }
    int V = 1;
    if (argc >= 12) {
        sscanf(argv[11], "%d", &V);
    }

    int dilation_h = 1;
    if (argc >= 13) {
        sscanf(argv[12], "%d", &dilation_h);
    }

    int dilation_w = 1;
    if (argc >= 14) {
        sscanf(argv[13], "%d", &dilation_w);
    }

    int P = floor((H + 2 * pad_h - dilation_h * (R - 1) - 1) / U + 1);
    int Q = floor((W + 2 * pad_w - dilation_w * (S - 1) - 1) / V + 1);
    int v_M = N * P * Q;
    int v_N = K;
    int v_K = C * R * S;
    double gflops = (2.0*v_M*v_N*v_K) * 1e-9;
    int bytes = 2 * (N * C * H * W + K * C * R * S + N * K * P * Q);
    double arithmetic_intensity = 2.0*v_M*v_N*v_K / bytes;
    double used_time = 0.0;
    int repeat = 10;

    printf("N: %d, H: %d, W: %d, C: %d, K: %d, R: %d, S: %d, pad_h: %d, pad_w: %d, U: %d, V: %d, dilation_h: %d, dilation_w: %d"
           "\n", 
            N, H, W, C, K, R, S, pad_h, pad_w, U, V, dilation_h, dilation_w);


    // printf("P: %d, Q: %d\n", P, Q);

    float * h_x = (float *)malloc(N * H * W * C * sizeof(float));
    float * h_w = (float *)malloc(K * R * S * C * sizeof(float));
    float * h_y = (float *)malloc(N * P * Q * K * sizeof(float));
    float * h_y1 = (float *)malloc(N * P * Q * K * sizeof(float));

    for (int j = 0; j < N * H * W * C; ++j) h_x[j] = static_cast<float>( 2*(rand() / double(RAND_MAX)) - 1 );
    for (int j = 0; j < K * R * S * C; ++j) h_w[j] = static_cast<float>( 2*(rand() / double(RAND_MAX)) - 1 );
    for (int j = 0; j < N * P * Q * K; ++j) h_y[j] = static_cast<float>(-1);
    for (int j = 0; j < N * P * Q * K; ++j) h_y1[j] = static_cast<float>(-1);



    bench::cc_implicit_gemm::conv2d_fwd(h_y, h_x, h_w, N, C, H, W,
                       K, P, Q, R, S, U, V, pad_h, pad_w, dilation_h, dilation_w, "zeros");
    // print_tensor(h_x, N, H, W, C);
    // print_tensor(h_w, K, C, R, S);
    // print_tensor(h_y, N, P, Q, K);
    used_time = 0.0;
    for (int i = 0; i < repeat; i++) {
        used_time += bench::cc_implicit_gemm::conv2d_fwd(h_y, h_x, h_w, N, C, H, W,
                       K, P, Q, R, S, U, V, pad_h, pad_w, dilation_h, dilation_w, "zeros");
    }
    used_time /= repeat;
    printf("cudnn_conv2d %dx%dx%dx%d %dx%dx%dx%d %dx%dx%dx%d, arithmetic_intensity:%f, im2col MNK: %dx%dx%d GFLOPs:%f, used_time: %fms, TFLOPS: %f\n", N, C, H, W,  
           K, C, R, S, N, K, P, Q, arithmetic_intensity, v_M, v_N, v_K, gflops, used_time, gflops/used_time);

}
