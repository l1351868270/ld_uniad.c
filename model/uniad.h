
#ifndef __LD_UNIAD_MODEL_UNIAD_H__
#define __LD_UNIAD_MODEL_UNIAD_H__

#include "config.h"
#include "img_backbone.h"
#include "run_state.h"

typedef struct {
    RunState * state;
    // PilotnetWeights weights;
    Config * config;
    ResNet * img_backbone;
} Uniad;

#endif // __LD_UNIAD_MODEL_UNIAD_H__