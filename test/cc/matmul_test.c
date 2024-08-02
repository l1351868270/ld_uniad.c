#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <assert.h>
#include "../../kernel/cc/matmul.h"
#include <cblas.h>

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

void print_tensor(float * tensor, int M, int N) {
    printf("[");
    for (int i = 0; i < M; i++) {
        printf("[");
        for (int j = 0; j < N; j++) {
            int offset = i * N + j;
            printf("%.3f, ", tensor[offset]);

        }
        printf("],\n");
    }
    printf("]\n");
}

// int equal_tensor(float * tensor1, float * tensor2, int N, int C, int H, int W) {
//     int equal = 1;
//     for (int n = 0; n < N; n++) {
//         for (int c = 0; c < C; c++) {
//             for (int h = 0; h < H; h++) {
//                 for (int w = 0; w < W; w++) {
//                     int offset = n * C * H * W + c * H * W + h * W + w;
//                     if (tensor1[offset] != tensor2[offset]) {
//                         printf("tensor1[%d, %d, %d, %d] = %.3f, tensor2[%d, %d, %d, %d] = %.3f\n", 
//                                n, c, h, w, tensor1[offset], n, c, h, w, tensor2[offset]);
//                         equal = 0;
//                     }
//                 }
//             }
//         }
//     }
//     return equal;
// }

int main(int argc, char ** argv) {
    time_t t = 0;
    srand(0);
    
    int M = 2227200;
    int N = 64;
    int K = 147;
    // int M = 2048;
    // int N = 2048;
    // int K = 2048;

    float * A;
    float * B;
    float * C;
    float * C2;

    posix_memalign((void **)&A, 128, M * K * sizeof(float));
    posix_memalign((void **)&B, 128, N * K * sizeof(float));
    posix_memalign((void **)&C, 128, M * N * sizeof(float));
    posix_memalign((void **)&C2, 128, M * N * sizeof(float));
    // size_t A_ptr = (size_t)A;
    printf("A: %p\n", A);
    printf("B: %p\n", B);
    printf("C: %p\n", C);
    printf("C2: %p\n", C2);
    
    // assert (&(A[0]) % 128 != 0);
    // assert (&B[0] % 128 != 0);
    // assert (&C[0] % 128 != 0);
    // assert (&C2[0] % 128 != 0);
    generate_tensor(A, M, K);
    generate_tensor(B, N, K);
    // print_tensor(A, 16, 16);
    // print_tensor(B, 16, 16);
    // matmul_fwd(C, A, B, M, N, K);
    // print_tensor(C, 16, 16);

    for (int i = 0; i < 1000; i++) {
        _blas_matmul_fwd(C2, A, B, M, N, K);
    }
    // _blas_matmul_fwd(C2, A, B, M, N, K);
    // print_tensor(C2, 16, 16);
    // _blas_matmul_fwd(C, A, B, M, N, K);
    
}