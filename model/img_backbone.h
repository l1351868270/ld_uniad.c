
#ifndef __LD_UNIAD_IMG_BACKBONE_H__
#define __LD_UNIAD_IMG_BACKBONE_H__

#include <stdio.h>
#include "config.h"

typedef struct {
    float * conv1;
    int   * conv1_meta;
    float * bn1_w;
    float * bn1_b;
    float * bn1_rm;
    float * bn1_rv;
    float * conv2;
    int   * conv2_meta;
    float * conv2_deform_w;
    float * conv2_deform_b;
    int   * conv2_deform_meta;
    float * bn2_w;
    float * bn2_b;
    float * bn2_rm;
    float * bn2_rv;
    float * conv3;
    int   * conv3_meta;
    float * bn3_w;
    float * bn3_b;
    float * bn3_rm;
    float * bn3_rv;
    float * downsample_conv;
    int   * downsample_conv_meta;
    float * downsample_bn_w;
    float * downsample_bn_b;
    float * downsample_bn_rm;
    float * downsample_bn_rv;

} Bottleneck;

typedef struct {
    float * conv1;
    int   * conv1_meta;
    float * bn1_w;
    float * bn1_b;
    float * bn1_rm;
    float * bn1_rv;
    int   * maxpool_meta;
    int num_stages;
    int stage_blocks[4];
    Bottleneck * bottleneck;
} ResNet;

int _img_backbone_get_layer_conv_size(int * meta, int num_stages, int * stage_blocks) {
    int offset = 0;
    int size = 0;
    for (int i = 0; i < num_stages; i++) {
        for (int j = 0; j < stage_blocks[i]; j++) {
            size += meta[offset] * meta[offset + 1] * meta[offset + 2] * meta[offset + 3];
            offset += 8;
        }
    }
    return size;
}

int _img_backbone_get_layer_conv_deform_size(int * meta, int num_stages, int * stage_blocks) {
    int offset = 0;
    int size = 0;
    for (int i = 2; i < num_stages; i++) {
        for (int j = 0; j < stage_blocks[i]; j++) {
            size += meta[offset] * meta[offset + 1] * meta[offset + 2] * meta[offset + 3];
            offset += 8;
        }
    }
    return size;
}

int _img_backbone_get_layer_deform_bias_size(int * meta, int num_stages, int * stage_blocks) {
    int offset = 0;
    int size = 0;
    for (int i = 2; i < num_stages; i++) {
        for (int j = 0; j < stage_blocks[i]; j++) {
            size += meta[offset];
            offset += 8;
        }
    }
    return size;
}

int _img_backbone_get_layer_bn_size(int * meta, int num_stages, int * stage_blocks) {
    int offset = 0;
    int size = 0;
    for (int i = 0; i < num_stages; i++) {
        for (int j = 0; j < stage_blocks[i]; j++) {
            size += meta[offset];
            offset += 8;
        }
    }
    return size;
}



void _img_backbone_load_conv_meta(int * meta, int num_stages, int * stage_blocks, char * name, FILE * file) {
    int rcount = 0;
    int meta_size = 0;
    for (int i = 0; i < num_stages; i++) {
        meta_size += stage_blocks[i] * 8;
    }

    rcount = fread(meta, sizeof(int), meta_size, file);
    if (rcount != meta_size) {
        fprintf(stderr, "Bad read _meta from model file\n");
        exit(1);
    }
#ifdef UNIAD_LOAD_WEIGHT_DEBUG
    int offset = 0;
    for (int i = 0; i < num_stages; i++) {
        printf("%s stage %d: \n", name, i);
        for (int j = 0; j < stage_blocks[i]; j++) {
            printf("(%d, %d, %d, %d, %d, %d, %d, %d) ", 
                   meta[offset + 0], meta[offset + 1], 
                   meta[offset + 2], meta[offset + 3],
                   meta[offset + 4], meta[offset + 5],
                   meta[offset + 6], meta[offset + 7]);
            offset += 8;
        }
        printf("\n");
    }
#endif // UNIAD_LOAD_WEIGHT_DEBUG
}

void _img_backbone_load_conv_deform_meta(int * meta, int num_stages, int * stage_blocks, char * name, FILE * file) {
    int rcount = 0;
    int meta_size = 0;
    for (int i = 2; i < num_stages; i++) {
        meta_size += stage_blocks[i] * 8;
    }

    rcount = fread(meta, sizeof(int), meta_size, file);
    if (rcount != meta_size) {
        fprintf(stderr, "Bad read _meta from model file\n");
        exit(1);
    }
#ifdef UNIAD_LOAD_WEIGHT_DEBUG
    int offset = 0;
    for (int i = 2; i < num_stages; i++) {
        printf("%s stage %d: \n", name, i);
        for (int j = 0; j < stage_blocks[i]; j++) {
            printf("(%d, %d, %d, %d, %d, %d, %d, %d) ", 
                   meta[offset + 0], meta[offset + 1], 
                   meta[offset + 2], meta[offset + 3],
                   meta[offset + 4], meta[offset + 5],
                   meta[offset + 6], meta[offset + 7]);
            offset += 8;
        }
        printf("\n");
    }
#endif // UNIAD_LOAD_WEIGHT_DEBUG
}


void _img_backbon_load_conv_bn(float * conv, float * bn_w, float * bn_b, float * bn_rm, float * bn_rv, int conv_size, int bn_size, FILE * file) {
    int rcount = 0;
    rcount = fread(conv, sizeof(float), conv_size, file);
    if (rcount != conv_size) {
        fprintf(stderr, "Bad read conv from model file\n");
        exit(1);
    }

    rcount = fread(bn_w, sizeof(float), bn_size, file);
    if (rcount != bn_size) {
        fprintf(stderr, "Bad read bn_w from model file\n");
        exit(1);
    }
    rcount = fread(bn_b, sizeof(float), bn_size, file);
    if (rcount != bn_size) {
        fprintf(stderr, "Bad read bn_b from model file\n");
        exit(1);
    }
    rcount = fread(bn_rm, sizeof(float), bn_size, file);
    if (rcount != bn_size) {
        fprintf(stderr, "Bad read bn_rm from model file\n");
        exit(1);
    }
    rcount = fread(bn_rv, sizeof(float), bn_size, file);
    if (rcount != bn_size) {
        fprintf(stderr, "Bad read bn_rv from model file\n");
        exit(1);
    }
}

void _img_backbon_load_conv_deform(float * conv_w, float * conv_b, int w_size, int b_size, FILE * file) {
    int rcount = 0;
    rcount = fread(conv_w, sizeof(float), w_size, file);
    if (rcount != w_size) {
        fprintf(stderr, "Bad read conv weight from model file\n");
        exit(1);
    }

    rcount = fread(conv_b, sizeof(float), b_size, file);
    if (rcount != b_size) {
        fprintf(stderr, "Bad read conv bias from model file\n");
        exit(1);
    }
}


void img_backbone_load_weights(ResNet * model, Config * config, FILE * file){
    model->num_stages = config->num_stages;
    model->stage_blocks[0] = config->stage_blocks[0];
    model->stage_blocks[1] = config->stage_blocks[1];
    model->stage_blocks[2] = config->stage_blocks[2];
    model->stage_blocks[3] = config->stage_blocks[3];
    
    int rcount = 0;
    model->conv1_meta = (int *)malloc(sizeof(int) * 8);
    rcount = fread(model->conv1_meta, sizeof(int), 8, file);
    if (rcount != 8) {
        fprintf(stderr, "Bad read conv1_meta from model file\n");
        exit(1);
    }
#ifdef UNIAD_LOAD_WEIGHT_DEBUG
    printf("model conv1_meta is: (%d, %d, %d, %d, %d, %d, %d, %d)\n", 
           model->conv1_meta[0], model->conv1_meta[1], model->conv1_meta[2], model->conv1_meta[3],
           model->conv1_meta[4], model->conv1_meta[5], model->conv1_meta[6], model->conv1_meta[7]);
#endif // UNIAD_LOAD_WEIGHT_DEBUG

    model->maxpool_meta = (int *)malloc(sizeof(int) * 4);
    rcount = fread(model->maxpool_meta, sizeof(int), 4, file);
    if (rcount != 4) {
        fprintf(stderr, "Bad read maxpool_meta from model file\n");
        exit(1);
    }

#ifdef UNIAD_LOAD_WEIGHT_DEBUG
    printf("model maxpool_meta is: (%d, %d, %d, %d)\n", 
           model->maxpool_meta[0], model->maxpool_meta[1], model->maxpool_meta[2], model->maxpool_meta[3]);
#endif // UNIAD_LOAD_WEIGHT_DEBUG

    model->bottleneck = (Bottleneck *)malloc(sizeof(Bottleneck));

    int meta_size = 0;
    for (int i = 0; i < config->num_stages; i++) {
        meta_size += config->stage_blocks[i] * 8;
    }

    model->bottleneck->conv1_meta = (int *)malloc(sizeof(int) * meta_size);
    _img_backbone_load_conv_meta(model->bottleneck->conv1_meta, config->num_stages, config->stage_blocks, "model->bottleneck->conv1_meta", file);
    model->bottleneck->conv2_meta = (int *)malloc(sizeof(int) * meta_size);
    _img_backbone_load_conv_meta(model->bottleneck->conv2_meta, config->num_stages, config->stage_blocks, "model->bottleneck->conv2_meta", file);
    int conv2_deform_meta_size = 0;
    for (int i = 2; i < config->num_stages; i++) {
        conv2_deform_meta_size += config->stage_blocks[i] * 8;
    }
    model->bottleneck->conv2_deform_meta = (int *)malloc(sizeof(int) * conv2_deform_meta_size);
    _img_backbone_load_conv_deform_meta(model->bottleneck->conv2_deform_meta, config->num_stages, config->stage_blocks, "model->bottleneck->conv2_deform_meta", file);
    model->bottleneck->conv3_meta = (int *)malloc(sizeof(int) * meta_size);
    _img_backbone_load_conv_meta(model->bottleneck->conv3_meta, config->num_stages, config->stage_blocks, "model->bottleneck->conv3_meta", file);
    
    int downsample_meta_size = config->num_stages * 8;
    model->bottleneck->downsample_conv_meta = (int *)malloc(sizeof(int) * downsample_meta_size);
    rcount = fread(model->bottleneck->downsample_conv_meta, sizeof(int), downsample_meta_size, file);
    if (rcount != downsample_meta_size) {
        fprintf(stderr, "Bad read res_layers->downsample_meta from model file\n");
        exit(1);
    }
    
#ifdef UNIAD_LOAD_WEIGHT_DEBUG
    int offset = 0;
    for (int i = 0; i < config->num_stages; i++) {
        printf("model->bottleneck->downsample_meta_size stage %d: ", i);
        printf("(%d, %d, %d, %d, %d, %d, %d, %d) ", 
                model->bottleneck->downsample_conv_meta[offset + 0], model->bottleneck->downsample_conv_meta[offset + 1], 
                model->bottleneck->downsample_conv_meta[offset + 2], model->bottleneck->downsample_conv_meta[offset + 3],
                model->bottleneck->downsample_conv_meta[offset + 4], model->bottleneck->downsample_conv_meta[offset + 5],
                model->bottleneck->downsample_conv_meta[offset + 6], model->bottleneck->downsample_conv_meta[offset + 7]);
        offset += 8;
        printf("\n");
    }
#endif // UNIAD_LOAD_WEIGHT_DEBUG

    // conv1
    int conv_size = model->conv1_meta[0] * model->conv1_meta[1] * model->conv1_meta[2] * model->conv1_meta[3];
    model->conv1 = (float *)malloc(sizeof(float) * conv_size);
    rcount = fread(model->conv1, sizeof(float), conv_size, file);
    if (rcount != conv_size) {
        fprintf(stderr, "Bad read conv1 from model file\n");
        exit(1);
    }

#ifdef UNIAD_LOAD_WEIGHT_CONV1_DEBUG
    printf("model conv1 is: (%d, %d, %d, %d)\n", model->conv1_meta[0], model->conv1_meta[1], model->conv1_meta[2], model->conv1_meta[3]);
    printf("[");
    for (int i = 0; i < model->conv1_meta[0]; i++) {
        printf("[");
        for (int j = 0; j < model->conv1_meta[1]; j++) {
            printf("[");
            for (int k = 0; k < model->conv1_meta[2]; k++) {
                printf("[");
                for (int x = 0; x < model->conv1_meta[3]; x++) {
                    int offset = i * model->conv1_meta[1] * model->conv1_meta[2] * model->conv1_meta[3] 
                               + j * model->conv1_meta[2] * model->conv1_meta[3] 
                               + k * model->conv1_meta[3] 
                               + x;
                    printf("%.4f ", model->conv1[offset]);
                }
                printf("],");
            }
            printf("],\n");
        }
        printf("],\n");
    }
    printf("]\n");
#endif // UNIAD_LOAD_WEIGHT_CONV1_DEBUG

    // bn1
    int bn_size = model->conv1_meta[0];
    model->bn1_w = (float *)malloc(sizeof(float) * bn_size);
    rcount = fread(model->bn1_w, sizeof(float), bn_size, file);
    if (rcount != bn_size) {
        fprintf(stderr, "Bad read bn1_w from model file\n");
        exit(1);
    }
    model->bn1_b = (float *)malloc(sizeof(float) * bn_size);
    rcount = fread(model->bn1_b, sizeof(float), bn_size, file);
    if (rcount != bn_size) {
        fprintf(stderr, "Bad read bn1_w from model file\n");
        exit(1);
    }
    model->bn1_rm = (float *)malloc(sizeof(float) * bn_size);
    rcount = fread(model->bn1_rm, sizeof(float), bn_size, file);
    if (rcount != bn_size) {
        fprintf(stderr, "Bad read bn1_w from model file\n");
        exit(1);
    }
    model->bn1_rv = (float *)malloc(sizeof(float) * bn_size);
    rcount = fread(model->bn1_rv, sizeof(float), bn_size, file);
    if (rcount != bn_size) {
        fprintf(stderr, "Bad read bn1_w from model file\n");
        exit(1);
    }
#ifdef UNIAD_LOAD_WEIGHT_BN1_DEBUG
    printf("model bn1_w is: (%d) \n", bn_size);
    printf("[");
    for (int i = 0; i < bn_size; i++) {
        printf("%.4f ", model->bn1_w[i]);
    }
    printf("]\n");
    printf("model bn1_b is: (%d) \n", bn_size);
    printf("[");
    for (int i = 0; i < bn_size; i++) {
        printf("%.4f ", model->bn1_b[i]);
    }
    printf("]\n");
    printf("model bn1_rm is: (%d) \n", bn_size);
    printf("[");
    for (int i = 0; i < bn_size; i++) {
        printf("%.4f ", model->bn1_rm[i]);
    }
    printf("]\n");
    printf("model bn1_rv is: (%d) \n", bn_size);
    printf("[");
    for (int i = 0; i < bn_size; i++) {
        printf("%.4f ", model->bn1_rv[i]);
    }
    printf("]\n");

#endif // UNIAD_LOAD_WEIGHT_BN1_DEBUG

    // layer conv1 bn1
    conv_size = _img_backbone_get_layer_conv_size(model->bottleneck->conv1_meta, config->num_stages, config->stage_blocks);
    model->bottleneck->conv1 = (float *)malloc(sizeof(float) * conv_size);
    bn_size = _img_backbone_get_layer_bn_size(model->bottleneck->conv1_meta, config->num_stages, config->stage_blocks);
    model->bottleneck->bn1_w = (float *)malloc(sizeof(float) * bn_size);
    model->bottleneck->bn1_b = (float *)malloc(sizeof(float) * bn_size);
    model->bottleneck->bn1_rm = (float *)malloc(sizeof(float) * bn_size);
    model->bottleneck->bn1_rv = (float *)malloc(sizeof(float) * bn_size);
    _img_backbon_load_conv_bn(model->bottleneck->conv1, model->bottleneck->bn1_w, model->bottleneck->bn1_b, 
                              model->bottleneck->bn1_rm, model->bottleneck->bn1_rv, conv_size, bn_size, file);

    // layer conv2 bn2
    conv_size = _img_backbone_get_layer_conv_size(model->bottleneck->conv2_meta, config->num_stages, config->stage_blocks);
    model->bottleneck->conv2 = (float *)malloc(sizeof(float) * conv_size);
    bn_size = _img_backbone_get_layer_bn_size(model->bottleneck->conv2_meta, config->num_stages, config->stage_blocks);
    model->bottleneck->bn2_w = (float *)malloc(sizeof(float) * bn_size);
    model->bottleneck->bn2_b = (float *)malloc(sizeof(float) * bn_size);
    model->bottleneck->bn2_rm = (float *)malloc(sizeof(float) * bn_size);
    model->bottleneck->bn2_rv = (float *)malloc(sizeof(float) * bn_size);
    _img_backbon_load_conv_bn(model->bottleneck->conv2, model->bottleneck->bn2_w, model->bottleneck->bn2_b, 
                              model->bottleneck->bn2_rm, model->bottleneck->bn2_rv, conv_size, bn_size, file);

    int weight_size = _img_backbone_get_layer_conv_deform_size(model->bottleneck->conv2_deform_meta, config->num_stages, config->stage_blocks);
    model->bottleneck->conv2_deform_w = (float *)malloc(sizeof(float) * weight_size);
    int bias_size = _img_backbone_get_layer_deform_bias_size(model->bottleneck->conv2_deform_meta, config->num_stages, config->stage_blocks);
    model->bottleneck->conv2_deform_b = (float *)malloc(sizeof(float) * bias_size);
    _img_backbon_load_conv_deform(model->bottleneck->conv2_deform_w, model->bottleneck->conv2_deform_b ,weight_size, bias_size, file);

    // layer conv3 bn3
    conv_size = _img_backbone_get_layer_conv_size(model->bottleneck->conv3_meta, config->num_stages, config->stage_blocks);
    model->bottleneck->conv3 = (float *)malloc(sizeof(float) * conv_size);
    bn_size = _img_backbone_get_layer_bn_size(model->bottleneck->conv3_meta, config->num_stages, config->stage_blocks);
    model->bottleneck->bn3_w = (float *)malloc(sizeof(float) * bn_size);
    model->bottleneck->bn3_b = (float *)malloc(sizeof(float) * bn_size);
    model->bottleneck->bn3_rm = (float *)malloc(sizeof(float) * bn_size);
    model->bottleneck->bn3_rv = (float *)malloc(sizeof(float) * bn_size);
    _img_backbon_load_conv_bn(model->bottleneck->conv3, model->bottleneck->bn3_w, model->bottleneck->bn3_b, 
                              model->bottleneck->bn3_rm, model->bottleneck->bn3_rv, conv_size, bn_size, file);

    // layer downsample conv bn
    int downsample_conv_size = 0;
    int downsample_bn_size = 0;
    int downsample_offset = 0;
    for (int i = 0; i < config->num_stages; i++) {
        downsample_conv_size += model->bottleneck->downsample_conv_meta[downsample_offset] *
                                model->bottleneck->downsample_conv_meta[downsample_offset + 1] *
                                model->bottleneck->downsample_conv_meta[downsample_offset + 2] *
                                model->bottleneck->downsample_conv_meta[downsample_offset + 3];
        downsample_bn_size += model->bottleneck->downsample_conv_meta[downsample_offset];
        downsample_offset += 8;
    }
    model->bottleneck->downsample_conv = (float *)malloc(sizeof(float) * downsample_conv_size);
    model->bottleneck->downsample_bn_w = (float *)malloc(sizeof(float) * downsample_bn_size);
    model->bottleneck->downsample_bn_b = (float *)malloc(sizeof(float) * downsample_bn_size);
    model->bottleneck->downsample_bn_rm = (float *)malloc(sizeof(float) * downsample_bn_size);
    model->bottleneck->downsample_bn_rv = (float *)malloc(sizeof(float) * downsample_bn_size);
    _img_backbon_load_conv_bn(model->bottleneck->downsample_conv, model->bottleneck->downsample_bn_w, model->bottleneck->downsample_bn_b, 
                              model->bottleneck->downsample_bn_rm, model->bottleneck->downsample_bn_rv, downsample_conv_size, downsample_bn_size, file);

#ifdef UNIAD_LOAD_WEIGHT_DOWNSAMPLE_BN_DEBUG
    printf("model downsample_bn_w is: (%d) \n", downsample_bn_size);
    printf("[");
    for (int i = 0; i < downsample_bn_size; i++) {
        printf("%.4f ", model->bottleneck->downsample_bn_w[i]);
    }
    printf("]\n");
    printf("model downsample_bn_b is: (%d) \n", downsample_bn_size);
    printf("[");
    for (int i = 0; i < downsample_bn_size; i++) {
        printf("%.4f ", model->bottleneck->downsample_bn_b[i]);
    }
    printf("]\n");
    printf("model downsample_bn_rm is: (%d) \n", downsample_bn_size);
    printf("[");
    for (int i = 0; i < downsample_bn_size; i++) {
        printf("%.4f ", model->bottleneck->downsample_bn_rm[i]);
    }
    printf("]\n");
    printf("model downsample_bn_rv is: (%d) \n", downsample_bn_size);
    printf("[");
    for (int i = 0; i < downsample_bn_size; i++) {
        printf("%.4f ", model->bottleneck->downsample_bn_rv[i]);
    }
    printf("]\n");

#endif // UNIAD_LOAD_WEIGHT_DOWNSAMPLE_BN_DEBUG

}

#endif // __LD_UNIAD_IMG_BACKBONE_H__