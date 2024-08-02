
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <math.h>
#include "ppm.h"
#include "kernel/cc/resize.h"
#include "kernel/cc/nhwc2nchw.h"
#include "kernel/cc/imnormalize.h"
#include "kernel/cc/impad_to_multiple.h"

#include "kernel/cc/conv2d.h"
#include "kernel/cc/batch_norm2d.h"
#include "kernel/cc/fuse_conv2d_batch_norm2d.h"
#include "kernel/cc/relu.h"
#include "kernel/cc/fuse_conv2d_batch_norm2d_relu.h"
#include "kernel/cc/maxpool2d.h"
#include "kernel/cc/identity.h"
#include "kernel/cc/add.h"
#include "kernel/cc/fuse_add_relu.h"

#include "model/img_backbone.h"
#include "model/config.h"
#include "model/run_state.h"
#include "model/uniad.h"


void load_weights(Uniad * model, const char * ckpt) {
    FILE *model_file = fopen(ckpt, "rb");
    if (model_file == NULL) {
        fprintf(stderr, "Error opening model file %s\n", ckpt);
        exit(1);
    }

    // Load config
    model->config = (Config *)malloc(sizeof(Config));
    load_config(model->config, model_file);
    // Load img_backbone
    model->img_backbone = (ResNet *)malloc(sizeof(ResNet));
    img_backbone_load_weights(model->img_backbone, model->config, model_file);
    
    // fclose(model_file);
}

void free_weights(Uniad * model) {
    free(model->state->identity);
    free(model->state->x);
    free(model->state->xb);
}

void resnet_fwd(ResNet * model, RunState * s, float * input, int * input_shape) {
    // conv1
    int _input_shape[4] = {input_shape[0], input_shape[1], input_shape[2], input_shape[3]};
    int _out_shape[4] = {};
    int out_channels = model->conv1_meta[0];
    int kernel_size = model->conv1_meta[4];
    int kernel_sizes[2] = {model->conv1_meta[4], model->conv1_meta[4]};
    int stride = model->conv1_meta[5];
    int strides[2] = {model->conv1_meta[5], model->conv1_meta[5]};
    int padding = model->conv1_meta[6];
    int paddings[2] = {model->conv1_meta[6], model->conv1_meta[6]};
    int dilation = model->conv1_meta[7];
    // int dilations[2] = {model->conv1_meta[7], model->conv1_meta[7]};
    conv2d_out_shape(_out_shape, _input_shape, out_channels, kernel_size, stride, padding, dilation);
    printf("_out_shape: (%d, %d, %d, %d)\n", _out_shape[0], _out_shape[1], _out_shape[2], _out_shape[3]);
    fuse_conv2d_batch_norm2d_relu_fwd(s->x, _out_shape, input, _input_shape, model->conv1, model->conv1_meta, NULL, strides, paddings, "zeros",
                                      model->bn1_w, model->bn1_b, model->bn1_rm, model->bn1_rv);
    // maxpool
    out_channels = _out_shape[1]; 
    kernel_size = model->maxpool_meta[0]; kernel_sizes[0] = model->maxpool_meta[0]; kernel_sizes[1] = model->maxpool_meta[0];
    stride = model->maxpool_meta[1]; strides[0] = model->maxpool_meta[1]; strides[1] = model->maxpool_meta[1];
    padding = model->maxpool_meta[2]; paddings[0] = model->maxpool_meta[2]; paddings[1] = model->maxpool_meta[2];
    dilation = model->maxpool_meta[3]; 
    _input_shape[0] = _out_shape[0]; _input_shape[1] = _out_shape[1]; _input_shape[2] = _out_shape[2]; _input_shape[3] = _out_shape[3];
    conv2d_out_shape(_out_shape, _input_shape, out_channels, kernel_size, stride, padding, dilation);
    printf("_out_shape: (%d, %d, %d, %d)\n", _out_shape[0], _out_shape[1], _out_shape[2], _out_shape[3]);
    maxpool2d_fwd(s->xb, _out_shape, s->x, _input_shape, kernel_sizes, strides, paddings, "zeros");

    int N, C, H, W;
    int num_stage = model->num_stages;
    int * stage_blocks = model->stage_blocks;
    int offset = 0;
    int downsample_offset = 0;
    int conv1_offset = 0;
    int bn1_offset = 0;
    int conv2_offset = 0;
    int bn2_offset = 0;
    int conv2_deform_meta_offset = 0;
    int conv2_deform_w_offset = 0;
    int conv2_deform_b_offset = 0;
    int conv3_offset = 0;
    int bn3_offset = 0;
    int downsample_conv_offset = 0;
    int downsample_bn_offset = 0;
    for (int i = 0; i < num_stage; i++) {
    // for (int i = 0; i < 2; i++) {
        int state_block = stage_blocks[i];
        for (int j = 0; j < state_block; j++) {
        // for (int j = 0; j < 1; j++) {
            N = _out_shape[0]; C = _out_shape[1]; H = _out_shape[2]; W = _out_shape[3];
            identity_fwd(s->identity, s->xb, _out_shape[0], _out_shape[1], _out_shape[2], _out_shape[3]);

            out_channels = model->bottleneck->conv1_meta[offset]; 
            kernel_size = model->bottleneck->conv1_meta[offset + 4]; kernel_sizes[0] = kernel_size; kernel_sizes[1] = kernel_size;
            stride = model->bottleneck->conv1_meta[offset + 5]; strides[0] = stride; strides[1] = stride;
            padding = model->bottleneck->conv1_meta[offset + 6]; paddings[0] = padding; paddings[1] = padding;
            dilation = model->bottleneck->conv1_meta[offset + 7]; 
            _input_shape[0] = _out_shape[0]; _input_shape[1] = _out_shape[1]; _input_shape[2] = _out_shape[2]; _input_shape[3] = _out_shape[3];
            conv2d_out_shape(_out_shape, _input_shape, out_channels, kernel_size, stride, padding, dilation);
            fuse_conv2d_batch_norm2d_relu_fwd(s->x, _out_shape, s->xb, _input_shape, model->bottleneck->conv1 + conv1_offset, 
                                              model->bottleneck->conv1_meta + offset, NULL, strides, paddings, "zeros",
                                              model->bottleneck->bn1_w + bn1_offset, model->bottleneck->bn1_b + bn1_offset, 
                                              model->bottleneck->bn1_rm + bn1_offset, model->bottleneck->bn1_rv + bn1_offset);
            conv1_offset += (model->bottleneck->conv1_meta + offset)[0] * (model->bottleneck->conv1_meta + offset)[1]
                           *(model->bottleneck->conv1_meta + offset)[2] * (model->bottleneck->conv1_meta + offset)[3];
            bn1_offset += (model->bottleneck->conv1_meta + offset)[0];

            if (i < 2) {
                out_channels = model->bottleneck->conv2_meta[offset]; 
                kernel_size = model->bottleneck->conv2_meta[offset + 4]; kernel_sizes[0] = kernel_size; kernel_sizes[1] = kernel_size;
                stride = model->bottleneck->conv2_meta[offset + 5]; strides[0] = stride; strides[1] = stride;
                padding = model->bottleneck->conv2_meta[offset + 6]; paddings[0] = padding; paddings[1] = padding;
                dilation = model->bottleneck->conv2_meta[offset + 7];
                _input_shape[0] = _out_shape[0]; _input_shape[1] = _out_shape[1]; _input_shape[2] = _out_shape[2]; _input_shape[3] = _out_shape[3];
                conv2d_out_shape(_out_shape, _input_shape, out_channels, kernel_size, stride, padding, dilation);
                fuse_conv2d_batch_norm2d_relu_fwd(s->xb, _out_shape, s->x, _input_shape, model->bottleneck->conv2 + conv2_offset, 
                                                model->bottleneck->conv2_meta + offset, NULL, strides, paddings, "zeros",
                                                model->bottleneck->bn2_w + bn2_offset, model->bottleneck->bn2_b + bn2_offset, 
                                                model->bottleneck->bn2_rm + bn2_offset, model->bottleneck->bn2_rv + bn2_offset);
                conv2_offset += (model->bottleneck->conv2_meta + offset)[0] * (model->bottleneck->conv2_meta + offset)[1]
                            *(model->bottleneck->conv2_meta + offset)[2] * (model->bottleneck->conv2_meta + offset)[3];
                bn2_offset += (model->bottleneck->conv2_meta + offset)[0];
            } else {
                out_channels = model->bottleneck->conv2_deform_meta[conv2_deform_meta_offset]; 
                kernel_size = model->bottleneck->conv2_deform_meta[conv2_deform_meta_offset + 4]; kernel_sizes[0] = kernel_size; kernel_sizes[1] = kernel_size;
                stride = model->bottleneck->conv2_deform_meta[conv2_deform_meta_offset + 5]; strides[0] = stride; strides[1] = stride;
                padding = model->bottleneck->conv2_deform_meta[conv2_deform_meta_offset + 6]; paddings[0] = padding; paddings[1] = padding;
                dilation = model->bottleneck->conv2_deform_meta[conv2_deform_meta_offset + 7];
                _input_shape[0] = _out_shape[0]; _input_shape[1] = _out_shape[1]; _input_shape[2] = _out_shape[2]; _input_shape[3] = _out_shape[3];
                conv2d_out_shape(_out_shape, _input_shape, out_channels, kernel_size, stride, padding, dilation);
                conv2d_fwd(s->xb, _out_shape, s->x, _input_shape, model->bottleneck->conv2_deform_w + conv2_deform_w_offset, 
                                model->bottleneck->conv2_deform_meta + conv2_deform_meta_offset, 
                                model->bottleneck->conv2_deform_b + conv2_deform_b_offset, strides, paddings, "zeros");
                conv2_deform_w_offset += (model->bottleneck->conv2_deform_meta + conv2_deform_meta_offset)[0]
                                       * (model->bottleneck->conv2_deform_meta + conv2_deform_meta_offset)[1]
                                       * (model->bottleneck->conv2_deform_meta + conv2_deform_meta_offset)[2]
                                       * (model->bottleneck->conv2_deform_meta + conv2_deform_meta_offset)[3];
                conv2_deform_b_offset += (model->bottleneck->conv2_deform_meta + conv2_deform_meta_offset)[0];
            }

            if (offset == 8 * 7) {
                exit(1);
            }

            out_channels = model->bottleneck->conv3_meta[offset]; 
            kernel_size = model->bottleneck->conv3_meta[offset + 4]; kernel_sizes[0] = kernel_size; kernel_sizes[1] = kernel_size;
            stride = model->bottleneck->conv3_meta[offset + 5]; strides[0] = stride; strides[1] = stride;
            padding = model->bottleneck->conv3_meta[offset + 6]; paddings[0] = padding; paddings[1] = padding;
            dilation = model->bottleneck->conv3_meta[offset + 7];
            _input_shape[0] = _out_shape[0]; _input_shape[1] = _out_shape[1]; _input_shape[2] = _out_shape[2]; _input_shape[3] = _out_shape[3];
            conv2d_out_shape(_out_shape, _input_shape, out_channels, kernel_size, stride, padding, dilation);
            fuse_conv2d_batch_norm2d_fwd(s->x, _out_shape, s->xb, _input_shape, model->bottleneck->conv3 + conv3_offset, 
                                         model->bottleneck->conv3_meta + offset, NULL, strides, paddings, "zeros",
                                         model->bottleneck->bn3_w + bn3_offset, model->bottleneck->bn3_b + bn3_offset, 
                                         model->bottleneck->bn3_rm + bn3_offset, model->bottleneck->bn3_rv + bn3_offset);
            conv3_offset += (model->bottleneck->conv3_meta + offset)[0] * (model->bottleneck->conv3_meta + offset)[1]
                           *(model->bottleneck->conv3_meta + offset)[2] * (model->bottleneck->conv3_meta + offset)[3];
            bn3_offset += (model->bottleneck->conv3_meta + offset)[0];



            if (j == 0) {
                out_channels = model->bottleneck->downsample_conv_meta[downsample_offset]; 
                kernel_size = model->bottleneck->downsample_conv_meta[downsample_offset + 4]; kernel_sizes[0] = kernel_size; kernel_sizes[1] = kernel_size;
                stride = model->bottleneck->downsample_conv_meta[downsample_offset + 5]; strides[0] = stride; strides[1] = stride;
                padding = model->bottleneck->downsample_conv_meta[downsample_offset + 6]; paddings[0] = padding; paddings[1] = padding;
                dilation = model->bottleneck->downsample_conv_meta[downsample_offset + 7];
                _input_shape[0] = N; _input_shape[1] = C; _input_shape[2] = H; _input_shape[3] = W;
                conv2d_out_shape(_out_shape, _input_shape, out_channels, kernel_size, stride, padding, dilation);
                fuse_conv2d_batch_norm2d_fwd(s->xb, _out_shape, s->identity, _input_shape, model->bottleneck->downsample_conv + downsample_conv_offset, 
                                             model->bottleneck->downsample_conv_meta + downsample_offset, NULL, strides, paddings, "zeros",
                                             model->bottleneck->downsample_bn_w + downsample_bn_offset, model->bottleneck->downsample_bn_b + downsample_bn_offset, 
                                             model->bottleneck->downsample_bn_rm + downsample_bn_offset, model->bottleneck->downsample_bn_rv + downsample_bn_offset);
                fuse_add_relu_fwd(s->xb, s->x, _out_shape[0], _out_shape[1], _out_shape[2], _out_shape[3]);
                downsample_conv_offset += (model->bottleneck->downsample_conv_meta + downsample_offset)[0] * (model->bottleneck->downsample_conv_meta + downsample_offset)[1]
                                         *(model->bottleneck->downsample_conv_meta + downsample_offset)[2] * (model->bottleneck->downsample_conv_meta + downsample_offset)[3];
                downsample_bn_offset += (model->bottleneck->downsample_conv_meta + downsample_offset)[0];
                downsample_offset += 8;
            } else {
                fuse_add_relu_fwd(s->x, s->identity, N, C, H, W);
                identity_fwd(s->xb, s->x, N, C, H, W);
            }
            // if (offset == 8 * 7) {
            //     exit(1);
            // }
            offset += 8;
        }
    }

}

void uniad_fwd(Uniad * model, float * input, int * input_shape) {
    RunState * s = model->state;
    ResNet * img_backbone = model->img_backbone;
    resnet_fwd(img_backbone, s, input, input_shape);

}

int main(int argc, char** argv) {
    char * ckpt = NULL;
    char * image_root = NULL;
    for (int i = 0; i < argc; i += 1) {
        if (strcmp(argv[i], "-ckpt") == 0) {
            ckpt = argv[i + 1];
        }

        if (strcmp(argv[i], "-image_root") == 0) {
            image_root = argv[i + 1];
        }

    }
#ifdef UNIAD_DEBUG
    printf("ckpt is: %s, image_root is: %s\n", ckpt, image_root);
#endif
    
    const int num_cam = 6;
    char * multiview_ppm_path[6] = {
        "/root/ld_uniad.c/tools/data/ld_nuscenes/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.ppm", 
        "/root/ld_uniad.c/tools/data/ld_nuscenes/samples/CAM_FRONT_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_RIGHT__1533151603520482.ppm",
        "/root/ld_uniad.c/tools/data/ld_nuscenes/samples/CAM_FRONT_LEFT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_LEFT__1533151603504799.ppm",
        "/root/ld_uniad.c/tools/data/ld_nuscenes/samples/CAM_BACK/n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151603537558.ppm",
        "/root/ld_uniad.c/tools/data/ld_nuscenes/samples/CAM_BACK_LEFT/n008-2018-08-01-15-16-36-0400__CAM_BACK_LEFT__1533151603547405.ppm", 
        "/root/ld_uniad.c/tools/data/ld_nuscenes/samples/CAM_BACK_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_BACK_RIGHT__1533151603528113.ppm"
    };

    MultiviewPPM multiview_ppm;
    multiview_ppm_read(&multiview_ppm, multiview_ppm_path, num_cam);
    multiview_ppm_rgb2bgr(&multiview_ppm);
    multiview_ppm_tofloat(&multiview_ppm);

    float mean[3] = {103.530, 116.280, 123.675};
    float std [3] = {1.0, 1.0, 1.0};
    imnormalize_fwd(multiview_ppm.fdata, mean, std, num_cam, multiview_ppm.height, multiview_ppm.width, 3);
    int size_divisor = 32;
    float pad_val = 0.0;
    int pad_h = ((multiview_ppm.height + size_divisor - 1) / size_divisor) * size_divisor;
    int pad_w = ((multiview_ppm.width + size_divisor - 1) / size_divisor) * size_divisor;
    float * x = (float *)malloc(sizeof(float) * multiview_ppm.num_cam * pad_h * pad_w * 3);
    float * xb = (float *)malloc(sizeof(float) * multiview_ppm.num_cam * pad_h * pad_w * 3);
    impad_to_multiple_fwd(x, multiview_ppm.fdata, multiview_ppm.num_cam, multiview_ppm.height, multiview_ppm.width, 3, size_divisor, pad_val);
    nhwc2nchw_fwd(xb, x, multiview_ppm.num_cam, pad_h, pad_w, 3);
    
    // Pilotnet model;
    // load_weights(&model, ckpt);
    // malloc_run_state(&model.state, &model.weights, x_shape);
    
    Uniad * model = (Uniad *)malloc(sizeof(Uniad));
    load_weights(model, ckpt);
    model->state = (RunState *)malloc(sizeof(RunState));
    int x_shape[4] = {multiview_ppm.num_cam, 3, pad_h, pad_w};
    run_state_init(model->state, model->img_backbone, x_shape);

    uniad_fwd(model, xb, x_shape);
    
    // PPM ppm;
    // char ppm_file[256];
    // int dst_rows = x_shape[1];
    // int dst_cols = x_shape[2];
    // float * x = (float *)malloc(sizeof(float) * dst_rows * dst_cols * x_shape[3]);
    // // for (int i = 0; i < 1; i++) {
    //     sprintf(ppm_file, "%s/%s" ,image_root, frame_path[0][0]);
    //     ppm_read(&ppm, ppm_file);
    //     ppm_rgb2bgr(&ppm);
    //     ppm_tofloat(&ppm);
    //     nhwc2nchw_fwd(x, ppm.fdata, 1, ppm.height, ppm.width, 3);

    //     resize_fwd(x, dst_rows, dst_cols, ppm.fdata, ppm.height, ppm.width);
    // }

    // for (int i = 0; i < 1000; i++) {
    //     sprintf(ppm_file, "%s/%d.ppm" ,image_path, i);
    //     ppm_read(&ppm, ppm_file);
    //     ppm_rgb2bgr(&ppm);
    //     ppm_normal(&ppm);
    //     float degrees = pilotnet_fwd(&model, dst_rows, dst_cols, ppm.fdata, ppm.height, ppm.width) * 180.0 / 3.14159265;
    //     printf("The %d frame predicted steering angle: %f degrees\n ", i, degrees);
    // }
    // free_ppm(&ppm);
}