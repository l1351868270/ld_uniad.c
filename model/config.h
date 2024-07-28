

#ifndef __LD_UNIAD_MODEL_CONFIG_H__
#define __LD_UNIAD_MODEL_CONFIG_H__
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int num_stages;
    int stage_blocks[4];
} Config;

void load_config(Config * config, FILE * file) {
    int * model_header_i = (int *)malloc(sizeof(int) * 256);
    int rcount = 0;

    rcount = fread(model_header_i, sizeof(int), 256, file);
    if (rcount != 256) {
        fprintf(stderr, "Bad read magic from model file\n");
        exit(1);
    }
    int magic = model_header_i[0];
    if (magic != 20240726) {
        fprintf(stderr, "Bad magic model file\n");
        exit(1);
    }

#ifdef UNIAD_LOAD_WEIGHT_DEBUG
    printf("model magic is: %d\n", magic);
#endif // UNIAD_LOAD_WEIGHT_DEBUG

    config->num_stages = model_header_i[1];
    for (int i = 0; i < config->num_stages; i++) {
        config->stage_blocks[i] = model_header_i[2 + i];
    }
#ifdef UNIAD_LOAD_WEIGHT_DEBUG
    printf("model config is: num_stages:%d, stage_blocks:(%d, %d, %d, %d)\n", config->num_stages, 
            config->stage_blocks[0], config->stage_blocks[1], config->stage_blocks[2], config->stage_blocks[3]);
#endif // UNIAD_LOAD_WEIGHT_DEBUG

}

#endif // __LD_UNIAD_MODEL_CONFIG_H__
