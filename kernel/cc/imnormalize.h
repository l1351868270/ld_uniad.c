
#ifndef __LD_UNIAD_IMNORMALIZE_H__
#define __LD_UNIAD_IMNORMALIZE_H__

#include <stdio.h>

void imnormalize_fwd(float * img, const float * mean, const float * std, const int N, const int H, const int W, const int C) {
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                for (int c = 0; c < C; c++) {
                    int offset = n * H * W * C + h * W * C + w * C + c;
                    img[offset] = (img[offset] - mean[c]) / std[c];
                }
            }
        }
    }
    
#ifdef UNIAD_IMNORMALIZE_DEBUG
    printf("imnormalize_fwd: (%d, %d, %d, %d)\n", N, H, W, C);
    printf("[");
    for (int n = 0; n < N; n++) {
        printf("[");
        for (int h = 0; h < H; h++) {
            printf("["); 
            for (int w = 0; w < W; w++) {
                printf("[");
                for (int c = 0; c < C; c++) {
                    int offset = n * H * W * C + h * W * C + w * C + c;
                    printf("%f, ", img[offset]);
                }
                printf("],");
            }
            printf("],\n");
        }
        printf("],\n");
    }
    printf("]\n");
#endif // UNIAD_IMNORMALIZE_DEBUG
}

#endif // __LD_UNIAD_IMNORMALIZE_H__
