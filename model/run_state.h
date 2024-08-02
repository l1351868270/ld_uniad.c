#ifndef __LD_UNIAD_MODEL_RUN_STATE_H__
#define __LD_UNIAD_MODEL_RUN_STATE_H__
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "img_backbone.h"
typedef struct {
    float * x;
    float * xb;
    float * identity;
} RunState;

// https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
void conv2d_out_shape(int * out_shape, int * in_shape, int out_channel, 
                     int kernel_size, int stride, int padding, int dilation) {
    int N    = in_shape[0];
    int H_in = in_shape[2];
    int W_in = in_shape[3];

    int C_out = out_channel;
    int H_out = floor((float)(H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / (float)stride + 1.0);
    int W_out = floor((float)(W_in + 2 * padding - dilation * (kernel_size - 1) - 1) / (float)stride + 1.0);

    out_shape[0] = N;
    out_shape[1] = C_out;
    out_shape[2] = H_out;
    out_shape[3] = W_out;
}

void run_state_init(RunState * state, ResNet * resnet, int * in_shape) {
    int _in_shape[4] = {in_shape[0], in_shape[1], in_shape[2], in_shape[3]};
    int _out_shape[4] = {};
    size_t x_size = _in_shape[0] * _in_shape[1] * _in_shape[2] * _in_shape[3];
    size_t max_size = x_size;

    int out_channels = resnet->conv1_meta[0];
    int kernel_size = resnet->conv1_meta[4];
    int stride = resnet->conv1_meta[5];
    int padding = resnet->conv1_meta[6];
    int dilation = resnet->conv1_meta[7];

    conv2d_out_shape(_out_shape, _in_shape, out_channels, kernel_size, stride, padding, dilation);
    x_size = _out_shape[0] * _out_shape[1] * _out_shape[2] * _out_shape[3];
    if (x_size > max_size) {
        max_size = x_size;
    }
    
    int num_stages = resnet->num_stages;
    int * stage_blocks = resnet->stage_blocks;

    int offset = 0;
    for (int i = 0; i < num_stages; i++) {
        int state_block = stage_blocks[i];
        for (int j = 0; j < state_block; j++) {
            out_channels = resnet->bottleneck->conv1_meta[offset];
            kernel_size  = resnet->bottleneck->conv1_meta[offset + 4];
            stride       = resnet->bottleneck->conv1_meta[offset + 5];
            padding      = resnet->bottleneck->conv1_meta[offset + 6];
            dilation     = resnet->bottleneck->conv1_meta[offset + 7];
            _in_shape[0] = _out_shape[0]; 
            _in_shape[1] = _out_shape[1]; _in_shape[2] = _out_shape[2]; _in_shape[3] = _out_shape[3];
            conv2d_out_shape(_out_shape, _in_shape, out_channels, kernel_size, stride, padding, dilation);
            x_size = _out_shape[0] * _out_shape[1] * _out_shape[2] * _out_shape[3];
            if (x_size > max_size) {
                max_size = x_size;
                printf("%d, %d, %d, %d\n", _out_shape[0], _out_shape[1], _out_shape[2], _out_shape[3]);
            }

            out_channels = resnet->bottleneck->conv2_meta[offset];
            kernel_size  = resnet->bottleneck->conv2_meta[offset + 4];
            stride       = resnet->bottleneck->conv2_meta[offset + 5];
            padding      = resnet->bottleneck->conv2_meta[offset + 6];
            dilation     = resnet->bottleneck->conv2_meta[offset + 7];
            _in_shape[0] = _out_shape[0]; _in_shape[1] = _out_shape[1]; _in_shape[2] = _out_shape[2]; _in_shape[3] = _out_shape[3];
            conv2d_out_shape(_out_shape, _in_shape, out_channels, kernel_size, stride, padding, dilation);
            x_size = _out_shape[0] * _out_shape[1] * _out_shape[2] * _out_shape[3];
            if (x_size > max_size) {
                max_size = x_size;
                printf("%d, %d, %d, %d\n", _out_shape[0], _out_shape[1], _out_shape[2], _out_shape[3]);
            }
            out_channels = resnet->bottleneck->conv3_meta[offset];
            kernel_size  = resnet->bottleneck->conv3_meta[offset + 4];
            stride       = resnet->bottleneck->conv3_meta[offset + 5];
            padding      = resnet->bottleneck->conv3_meta[offset + 6];
            dilation     = resnet->bottleneck->conv3_meta[offset + 7];
            _in_shape[0] = _out_shape[0]; _in_shape[1] = _out_shape[1]; _in_shape[2] = _out_shape[2]; _in_shape[3] = _out_shape[3];
            conv2d_out_shape(_out_shape, _in_shape, out_channels, kernel_size, stride, padding, dilation);
            x_size = _out_shape[0] * _out_shape[1] * _out_shape[2] * _out_shape[3];
            if (x_size > max_size) {
                max_size = x_size;
                printf("%d, %d, %d, %d\n", _out_shape[0], _out_shape[1], _out_shape[2], _out_shape[3]);
            }
            offset += 8;
        }
    }

    state->x = (float *)malloc(sizeof(float) * max_size);
    state->xb = (float *)malloc(sizeof(float) * max_size);
    state->identity = (float *)malloc(sizeof(float) * max_size);

#if defined(UNIAD_RUNSTATE_DEBUG) || defined(UNIAD_DEBUG)
    printf("[run_state_init]: max_size is: %lu, max memory alloc is: %lubytes, %.2fKB, %.2fMB, %.2fGB\n", 
            max_size, max_size * sizeof(float), max_size * sizeof(float) / 1024.0, 
            max_size * sizeof(float) / 1024.0 / 1024.0, 
            max_size * sizeof(float) / 1024.0 / 1024.0 / 1024.0);
#endif // UNIAD_RUNSTATE_DEBUG   

}

#endif // __LD_UNIAD_MODEL_RUN_STATE_H__