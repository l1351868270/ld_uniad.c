
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include "../../kernel/cc/im2col.h"
#include "../../kernel/cc/col2im.h"

float frand() {
    return (float)rand() / (float)RAND_MAX;
}

void generate_tensor(float * tensor, int N, int C, int H, int W) {
    for (int i = 0; i < N * C * H * W; i++) {
        tensor[i] = frand();
    }
}

void generate_range_tensor(float * tensor, int N, int C, int H, int W) {
    for (int i = 0; i < N * C * H * W; i++) {
        tensor[i] = i;
    }
}

void print_tensor(float * tensor, int N, int C, int H, int W) {
    printf("[");
    for (int n = 0; n < N; n++) {
        printf("[");
        for (int c = 0; c < C; c++) {
            printf("[");
            for (int h = 0; h < H; h++) {
                printf("[");
                for (int w = 0; w < W; w++) {
                    int offset = n * C * H * W + c * H * W + h * W + w;
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

int equal_tensor(float * tensor1, float * tensor2, int N, int C, int H, int W) {
    int equal = 1;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    int offset = n * C * H * W + c * H * W + h * W + w;
                    if (tensor1[offset] != tensor2[offset]) {
                        printf("tensor1[%d, %d, %d, %d] = %.3f, tensor2[%d, %d, %d, %d] = %.3f\n", 
                               n, c, h, w, tensor1[offset], n, c, h, w, tensor2[offset]);
                        equal = 0;
                    }
                }
            }
        }
    }
    return equal;
}

int main(int argc, char ** argv) {
    time_t t = 0;
    srand(0);

    int N = 6;
    int C_in = 3;
    int H_in = 928;
    int W_in = 1600;

    int C_out = 64;
    int kernel_sizes[2] = {7, 7};
    int strides[2] = {2, 2};
    int paddings[2] = {3, 3};
    int dilations[2] = {1, 1};

    // int N = 6;
    // int C_in = 3;
    // int H_in = 16;
    // int W_in = 16;

    // int C_out = 5;
    // int kernel_sizes[2] = {7, 7};
    // int strides[2] = {2, 2};
    // int paddings[2] = {3, 3};
    // int dilations[2] = {1, 1};


    int H_out = floor((float)(H_in + 2 * paddings[0] - dilations[0] * (kernel_sizes[0] - 1) - 1) / (float)strides[0] + 1.0);
    int W_out = floor((float)(W_in + 2 * paddings[1] - dilations[1] * (kernel_sizes[1] - 1) - 1) / (float)strides[1] + 1.0);
    
    int im2col_rows = N * H_out * W_out;
    int im2col_cols = C_in * kernel_sizes[0] * kernel_sizes[1];

    float * data_im  = (float *)malloc(N * C_in * H_in * W_in * sizeof(float));
    float * data_col = (float *)malloc(im2col_rows * im2col_cols * sizeof(float));
    generate_range_tensor(data_im, N, C_in, H_in, W_in);
    // print_tensor(data_im, N, C_in, H_in, W_in);
    im2col_fwd(data_col, data_im, N, C_in, H_in, W_in, kernel_sizes, strides, paddings, dilations, "zeros");
    float * data_im2 = (float *)malloc(N * C_in * H_in * W_in * sizeof(float));
    col2im_fwd(data_im2, data_col, N, C_in, H_in, W_in, kernel_sizes, strides, paddings, dilations, "zeros");
    assert (equal_tensor(data_im, data_im2, N, C_in, H_in, W_in));
    printf("im2col_rows:%d, im2col_cols:%d, (%d, %d)\n", im2col_rows, im2col_cols, H_out, W_out);

    im2col_v1_fwd(data_col, data_im, N, C_in, H_in, W_in, kernel_sizes, strides, paddings, dilations, "zeros");
    assert (equal_tensor(data_im, data_im2, N, C_in, H_in, W_in));
    return 0;
}
