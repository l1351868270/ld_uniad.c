#ifndef __LD_UNIAD_IMPAD_TO_MULTIPLE_H__
#define __LD_UNIAD_IMPAD_TO_MULTIPLE_H__

#include <stdio.h>
// https://github.com/open-mmlab/mmcv/blob/main/mmcv/image/geometric.py
void impad_to_multiple_fwd(float * dst, float * src, int N, int H, int W, int C, int divisor, float pad_val) {
    int pad_h = ((H + divisor - 1) / divisor) * divisor;
    int pad_w = ((W + divisor - 1) / divisor) * divisor;
    int padding[4] = {0, 0, pad_w - W, pad_h - H}; // left, top, right, bottom
    int top = padding[1];
    int bottom = padding[3];
    int left = padding[0];
    int right = padding[2];

    for (int n = 0; n < N; n++) {
        for (int h = 0; h < pad_h; h++) {
            for (int w = 0; w < pad_w; w++) {
                for (int c = 0; c < C; c++) {
                    int dst_offset = n * pad_h * pad_w * C + h * pad_w * C + w * C + c;
                    int src_offset = n * H * W * C         + h * W * C     + w * C + c;
                    if (h < H && w < W) {
                        dst[dst_offset] = src[src_offset];
                    } else {
                        dst[dst_offset] = pad_val;
                    }
                }
            }
        }
    }

#ifdef UNIAD_IMPAD_TO_MULTIPLE_DEBUG
    printf("impad_to_multiple_fwd: (%d, %d, %d, %d) (%d, %d)\n", N, pad_h, pad_w, C, H, W);
    printf("[");
    for (int n = 0; n < N; n++) {
        printf("[");
        for (int h = 0; h < H; h++) {
            printf("["); 
            for (int w = 0; w < W; w++) {
                printf("[");
                for (int c = 0; c < C; c++) {
                    int offset = n * H * W * C + h * W * C + w * C + c;
                    printf("%.2f, ", dst[offset]);
                }
                printf("],");
            }
            printf("],\n");
        }
        printf("],\n");
    }
    printf("]\n");
#endif // UNIAD_NHWC2NCHW_DEBUG

}

#endif // __LD_UNIAD_IMPAD_TO_MULTIPLE_H__