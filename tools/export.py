'''
Adapted from https://github.com/OpenDriveLab/UniAD/blob/main/tools/test.py
'''


import argparse
import os
import struct
import numpy as np
from mmcv import Config, DictAction
from mmdet.datasets import replace_ImageToTensor
import mmdet.models
from mmdet3d.models import build_model
import mmcv
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
import mmcv.model_zoo
import mmdet3d.models

from export_img_backbone import fp32_write_img_backbone

def write_header(config, file):
    model_header_i = np.zeros(256, dtype=np.int32)
    model_header_i[0] = 20240726
    model_header_i[1] = 4 # resnet num_stages       
    model_header_i[2:6] = (3, 4, 23, 3) # resnet arch_settings    
    # print(f"model_header_i: {model_header_i}")

    file.write(model_header_i.tobytes())

def fp32_write_model(model, cfg, filename):
    print(f"write model to {filename}, the keys of model is {len(model.state_dict())}")
    
    with open(filename, "wb") as file:
        write_header(cfg, file)
        sd = model.state_dict()

        fp32_write_img_backbone(sd, cfg['model'], file)
        
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path', type=str, default='configs/stage2_e2e/base_e2e.py')
    parser.add_argument('checkpoint', help='checkpoint file', type=str, default='work_dirs/stage2_e2e/latest.pth')
    parser.add_argument("--filepath", type=str, default="uniad_base_e2e.bin")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
    
    cfg.model.pretrained = None

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    cfg.model.train_cfg = None

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    sd = model.state_dict()
    # print(sd.keys())
    # print(f"cfg: {cfg}")
    fp32_write_model(model, cfg, args.filepath)
    
if __name__ == '__main__':
    main()