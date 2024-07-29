
#ifndef __LD_UNIAD_RESIZE_H__
#define __LD_UNIAD_RESIZE_H__

#include <stdio.h>
#include <math.h>

// Adapted from https://github.com/opencv/opencv/blob/master/modules/imgproc/src/resize.cpp
void resize_fwd(float * dst, const int dst_rows, const int dst_cols, 
          const float * src, const int src_rows, const int src_cols) {
    printf("resize_fwd： (%d, %d, %d, %d)\n", dst_rows, dst_cols, src_rows, src_cols);
    double inv_scale_x = ((double)dst_cols)/((double)src_cols);
    double inv_scale_y = ((double)dst_rows)/((double)src_rows);

    double scale_x = 1. / inv_scale_x;
    double scale_y = 1. / inv_scale_y;

    int k, sx, sy, dx, dy;
    float fx, fy;

    for (dy = 0; dy < dst_rows; dy++) {
        
        for (dx = 0; dx < dst_cols; dx++) {
            if (scale_x > 1.0) {
                fx = (float)((dx+0.5)*scale_x - 0.5);
            } else {
                fx = (float)(dx * scale_x);
            }
            
            sx = floor(fx);
            fx -= sx;
            if (scale_y > 1.0) {
                fy = (float)((dy+0.5)*scale_y - 0.5);
            } else {
                fy = (float)(dy * scale_y);
            }

            sy = floor(fy);
            fy -= sy;
            
            if (sx < 0 || sx >= src_cols) {
                fprintf(stderr, "sx must between: [0, %d), but is %d\n", dst_cols, sx);
            }

            if (sy < 0 || sy >= src_rows) {
                fprintf(stderr, "sy must between: [0, %d), but is %d %d, %f %d, %d %f,\n", dst_rows, sy, dy, scale_y, dst_rows, src_rows, inv_scale_y);
            }

            int offset_src_lt = sy * src_cols * 3 + sx * 3; // left-top
            int offset_src_rt = sy * src_cols * 3 + (sx + 1) * 3; // right-top
            int offset_src_ld = (sy + 1) * src_cols * 3 + sx * 3; // left-down
            int offset_src_rd = (sy + 1) * src_cols * 3 + (sx + 1) * 3; // right-down
             
            int offset_dst = dy * dst_cols * 3 + dx * 3;
            
            for (int c = 0; c < 3; c++) {
                float src_lt = src[offset_src_lt + c];
                float src_rt = sx < src_cols ? src[offset_src_rt + c] : 0.0;
                float src_ld = sy < src_rows ? src[offset_src_ld + c] : 0.0;
                float src_rd = sx < src_cols && sy < src_rows ? src[offset_src_rd + c] : 0.0;
                float f_E = src_lt * (1 - fx) + src_rt * fx;
                float f_F = src_ld * (1 - fx) + src_rd * fx;
                float target = f_E * (1. - fy) + f_F * fy;
                dst[offset_dst + c] = target;
            }
        }
    }

#ifdef UNIAD_RESIZE_DEBUG

    printf("ppm_resize src: \n");
    printf("[");
    for (int i = 0; i < src_rows; i++) {
        printf("[");
        for (int j = 0; j < src_cols; j++) {
            int offset = i * src_cols * 3 + j * 3;
            printf("[%f, %f, %f]", src[offset], src[offset + 1], src[offset + 2]);
        }
        printf("]\n");
    }
    printf("], shape=(%d, %d, 3)\n", src_rows, src_cols);

    printf("ppm_resize dst: \n");
    printf("[");
    for (int i = 0; i < dst_rows; i++) {
        printf("[");
        for (int j = 0; j < dst_cols; j++) {
            int offset = i * dst_cols * 3 + j * 3;
            printf("[%f, %f, %f]", dst[offset], dst[offset + 1], dst[offset + 2]);
        }
        printf("]\n");
    }
    printf("], shape=(%d, %d, 3)\n", dst_rows, dst_cols);
#endif // UNIAD_RESIZE_DEBUG

}

#endif // __LD_UNIAD_RESIZE_H__