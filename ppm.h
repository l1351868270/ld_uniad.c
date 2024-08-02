
#ifndef __LD_UNIAD_PPM_H__
#define __LD_UNIAD_PPM_H__
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

enum PPM_DTYPE{
    PPM_UINT8 = 0,
    PPM_FLOAT32
};

typedef struct {
    int height;
    int width;
    enum PPM_DTYPE dtype;
    // data fdata use a same memory
    unsigned char *data;
    float * fdata;
} PPM;



void ppm_read(PPM * ppm, const char * ppm_path) {
    FILE * file = fopen(ppm_path, "rb");
    if (file == NULL) {
        fprintf(stderr, "Bad read ppm: %s\n", ppm_path);
        exit(1);
    }

    char format[3];
    int rcount = 0;
    rcount = fscanf(file, "%s", format);
    if (rcount == EOF) {
        fprintf(stderr, "fscanf format err");
    }
    fgetc(file);
    int width, height, max_value;
    rcount = fscanf(file, "%d %d", &width, &height);
    if (rcount == EOF) {
        fprintf(stderr, "fscanf width height err");
    }
    fgetc(file);
    rcount = fscanf(file, "%d", &max_value);
    if (rcount == EOF) {
        fprintf(stderr, "fscanf max_value err");
    }
    fgetc(file);

    ppm->width = width;
    ppm->height = height;
    ppm->dtype = PPM_UINT8;

    ppm->data = (unsigned char *)malloc(width * height * 3 * sizeof(float));
    rcount = fread(ppm->data, sizeof(unsigned char), width * height * 3, file);
    if (rcount != width * height * 3) {
        fprintf(stderr, "fread ppm data");
    }
#ifdef PPM_READ_DEBUG
    printf("format: %s\n", format);
    printf("width: %d\n",  width);
    printf("height: %d\n",  height);
    printf("max_value: %d\n", max_value);

    printf("ppm_read: \n");
    printf("[");
    for (int i = 0; i < height; i++) {
        printf("[");
        for (int j = 0; j < width; j++) {
            int offset = i * width * 3 + j * 3;
            printf("[%d, %d, %d]", ppm->data[offset], ppm->data[offset + 1], ppm->data[offset + 2]);
        }
        printf("]\n");
    }
    printf("], shape=(%d, %d, 3)\n", height, width);
#endif // PPM_READ_DEBUG
}

void ppm_rgb2bgr(PPM * ppm) {
    int height = ppm->height;
    int width = ppm->width;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int offset = i * width * 3 + j * 3;
            unsigned char tmp = ppm->data[offset];
            unsigned char tmp1 = ppm->data[offset+2];
            ppm->data[offset] = tmp1;
            ppm->data[offset + 2] = tmp;
        }
    }

#ifdef PPM_RGB2BGR_DEBUG
    printf("[");
    for (int i = 0; i < height; i++) {
        printf("[");
        for (int j = 0; j < width; j++) {
            int offset = i * width * 3 + j * 3;
            printf("[%d, %d, %d]", ppm->data[offset], ppm->data[offset + 1], ppm->data[offset + 2]);
        }
        printf("]\n");
    }
    printf("], shape=(%d, %d, 3)\n", height, width);
#endif // PPM_RGB2BGR_DEBUG
}

void ppm_tofloat(PPM * ppm) {
    int height = ppm->height;
    int width = ppm->width;
    ppm->dtype = PPM_FLOAT32;
    int size = height * width * 3;

    ppm->fdata = (float*)ppm->data;

    for (int i = size-1; i >=0; i--) {
        ppm->fdata[i] = (float)ppm->data[i];
    }

#ifdef PPM_TOFLOAT_DEBUG
    printf("ppm_tofloat: \n");
    printf("[");
    for (int i = 0; i < height; i++) {
        printf("[");
        for (int j = 0; j < width; j++) {
            int offset = i * width * 3 + j * 3;
            printf("[%f, %f, %f]", ppm->fdata[offset], ppm->fdata[offset + 1], ppm->fdata[offset + 2]);
        }
        printf("]\n");
    }
    printf("], shape=(%d, %d, 3)\n", height, width);
#endif // PPM_TOFLOAT_DEBUG
}

// Adapted from https://github.com/opencv/opencv/blob/master/modules/imgproc/src/resize.cpp
void ppm_resize(float * dst, const int dst_rows, const int dst_cols, 
          const float * src, const int src_rows, const int src_cols) {
    double inv_scale_x = ((double)dst_cols)/((double)src_cols);
    double inv_scale_y = ((double)dst_rows)/((double)src_rows);

    double scale_x = 1. / inv_scale_x;
    double scale_y = 1. / inv_scale_y;

    int sx, sy, dx, dy;
    float fx, fy;

    for (dy = 0; dy < dst_rows; dy++) {
        
        for (dx = 0; dx < dst_cols; dx++) {
            fx = (float)((dx+0.5)*scale_x - 0.5);
            sx = floor(fx);
            fx -= sx;

            fy = (float)((dy+0.5)*scale_y - 0.5);
            sy = floor(fy);
            fy -= sy;
            
            if (sx < 0 || sx >= src_cols) {
                fprintf(stderr, "sx must between: [0, %d), but is %d\n", dst_cols, sx);
            }

            if (sy < 0 || sy >= src_rows) {
                fprintf(stderr, "sy must between: [0, %d), but is %d\n", dst_rows, sy);
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

void free_ppm(PPM * ppm) {
    free(ppm->data);
}


typedef struct {
    int num_cam;
    int height;
    int width;
    enum PPM_DTYPE dtype;
    // data fdata use a same memory
    unsigned char *data;
    float * fdata;
} MultiviewPPM;

void multiview_ppm_read(MultiviewPPM * multiview_ppm, char ** multiview_ppm_path, const int num_cam) {
    multiview_ppm->num_cam = num_cam;
    for (int i = 0; i < num_cam; i++) {
        const char * ppm_path = multiview_ppm_path[i];
        FILE * file = fopen(ppm_path, "rb");
        if (file == NULL) {
            fprintf(stderr, "Bad read ppm: %s\n", ppm_path);
            exit(1);
        }

        char format[3];
        int rcount = 0;
        rcount = fscanf(file, "%s", format);
        if (rcount == EOF) {
            fprintf(stderr, "[multiview_ppm_read]: fscanf format err");
        }
        fgetc(file);
        int width, height, max_value;
        rcount = fscanf(file, "%d %d", &width, &height);
        if (rcount == EOF) {
            fprintf(stderr, "[multiview_ppm_read]: fscanf width height err");
        }
        fgetc(file);
        rcount = fscanf(file, "%d", &max_value);
        if (rcount == EOF) {
            fprintf(stderr, "[multiview_ppm_read]: fscanf max_value err");
        }
        fgetc(file);
        if (i == 0) {
            multiview_ppm->width = width;
            multiview_ppm->height = height;
            multiview_ppm->dtype = PPM_UINT8;
            multiview_ppm->data = (unsigned char *)malloc(num_cam * width * height * 3 * sizeof(float));
        } else {
            assert(multiview_ppm->width == width);
            assert(multiview_ppm->height == height);
        }

        rcount = fread(multiview_ppm->data + i * width * height * 3, sizeof(unsigned char), width * height * 3, file);
        if (rcount != width * height * 3) {
            fprintf(stderr, "[multiview_ppm_read]: fread data error\n");
        }
    } 
    
#ifdef MULTIVIEW_PPM_READ_DEBUG
    printf("num_cam: %D\n", multiview_ppm->num_cam);
    printf("width: %d\n",  multiview_ppm->width);
    printf("height: %d\n",  multiview_ppm->height);

    printf("multiview_ppm_read: (%d, %d, %d, 3)\n", multiview_ppm->num_cam, multiview_ppm->width, multiview_ppm->height);
    printf("[");
    for (int i = 0; i < num_cam; i++) {
        printf("[");
        for (int j = 0; j < multiview_ppm->height; j++) {
            printf("[");
            for (int k = 0; k < multiview_ppm->width; k++) {
                int offset = i * multiview_ppm->width * multiview_ppm->height * 3 + j * multiview_ppm->width * 3 + k * 3;
                printf("[%d, %d, %d]", multiview_ppm->data[offset], multiview_ppm->data[offset + 1], multiview_ppm->data[offset + 2]);
            }
            printf("],\n");
        }
        printf("],\n");
    }
    printf("]\n");
#endif // MULTIVIEW_PPM_READ_DEBUG
}

void multiview_ppm_rgb2bgr(MultiviewPPM * multiview_ppm) {
    int num_cam = multiview_ppm->num_cam;
    int height = multiview_ppm->height;
    int width = multiview_ppm->width;
    
    for (int i = 0; i < num_cam; i++) {
        for (int j = 0; j < height; j++) {
            for (int k = 0; k < width; k++) {
                int offset = i * height * width * 3 + j * width * 3 + k * 3;
                unsigned char tmp = multiview_ppm->data[offset];
                unsigned char tmp1 = multiview_ppm->data[offset+2];
                multiview_ppm->data[offset] = tmp1;
                multiview_ppm->data[offset + 2] = tmp;
            
            }
        }
    }

#ifdef MULTIVIEW_PPM_RGB2BGR_DEBUG
    printf("multiview_ppm_rgb2bgr: (%d, %d, %d, 3)\n", num_cam, height, width);
    printf("[");
    for (int i = 0; i < num_cam; i++) {
        printf("[");
        for (int j = 0; j < height; j++) {
            printf("[");
            for (int k = 0; k < width; k++) {
                int offset = i * height * width * 3 + j * width * 3 + k * 3;
                printf("[%d, %d, %d]", multiview_ppm->data[offset], multiview_ppm->data[offset + 1], multiview_ppm->data[offset + 2]);
            }
            printf("],\n");
        }
        printf("],\n");
    }
    printf("]\n");
#endif // MULTIVIEW_PPM_RGB2BGR_DEBUG
}

void multiview_ppm_tofloat(MultiviewPPM * multiview_ppm) {
    int num_cam = multiview_ppm->num_cam;
    int height = multiview_ppm->height;
    int width = multiview_ppm->width;
    multiview_ppm->dtype = PPM_FLOAT32;
    int size = num_cam * height * width * 3;

    multiview_ppm->fdata = (float*)multiview_ppm->data;

    for (int i = size-1; i >=0; i--) {
        multiview_ppm->fdata[i] = (float)multiview_ppm->data[i];
    }

#ifdef MULTIVIEW_PPM_TOFLOAT_DEBUG
    printf("multiview_ppm_tofloat: (%d, %d, %d, 3)\n", num_cam, height, width);
    printf("[");
    for (int i = 0; i < num_cam; i++) {
        printf("[");
        for (int j = 0; j < height; j++) {
            printf("[");
            for (int k = 0; k < width; k++) {
                int offset = i * height * width * 3 + j * width * 3 + k * 3;
                printf("[%f, %f, %f]", multiview_ppm->fdata[offset], multiview_ppm->fdata[offset + 1], multiview_ppm->fdata[offset + 2]);
            }
            printf("],\n");
        }
        printf("],\n");
    }
    printf("]\n");
#endif // MULTIVIEW_PPM_TOFLOAT_DEBUG
}


typedef struct {
    int num_cam;
    int height;
    int width;
    enum PPM_DTYPE dtype;
    unsigned char *data;
    float * fdata;
} Frame;


void read_frame(Frame * frame, PPM * ppm, const char ** ppm_path) {
    assert(frame->height == ppm->height);
    for (int i = 0; i < frame->num_cam; i++) {
        ppm_read(ppm, ppm_path[i]);
        frame->data = ppm->data;
    }
}


void free_frame(Frame * frame) {
    free(frame->data);
}


#endif