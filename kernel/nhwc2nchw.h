#ifndef __LD_UNIAD_NHWC2NCHW_H__
#define __LD_UNIAD_NHWC2NCHW_H__

#include <stdio.h>

void nhwc2nchw_fwd(float * dst, const float * src, const int N, const int H, const int W, const int C) {
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    int dst_offset = n * C * H * W + c * H * W + h * W + w;
                    int src_offset = n * H * W * C + h * W * C + w * C + c;
                    dst[dst_offset] = src[src_offset];
                }
            }
        }
    }
#ifdef UNIAD_NHWC2NCHW_DEBUG
    printf("nhwc2nchw_fwd: (%d, %d, %d, %d)\n", N, C, H, W);
    printf("[");
    for (int n = 0; n < N; n++) {
        printf("[");
        for (int c = 0; c < C; c++) {
            printf("[");
            // for (int h = 0; h < H; h++) {
            for (int h = 0; h < 16; h++) {
                printf("[");
                // for (int w = 0; w < W; w++) {
                for (int w = 0; w < 16; w++) {
                    int dst_offset = n * C * H * W + c * H * W + h * W + w;
                    printf("%.2f, ", dst[dst_offset]);
                }
                printf("],\n");
            }
            printf("],\n");
        }
        printf("],\n");
    }
    printf("]\n");
#endif // UNIAD_NHWC2NCHW_DEBUG

}

#endif // __LD_UNIAD_NHWC2NCHW_H__