
import struct

from export_utils import write_fp32, MODEL_LIANMENT

def fp32_write_bottleneck(model_sd, cfg, file):
    pass

def fp32_write_img_backbone_layer(model_sd, format_str, stage_blocks, file):
    s = 1
    for i, num_blocks in enumerate(stage_blocks):
        for j in range(num_blocks):
            for ms in model_sd[format_str.format(i+1, j)].shape:
                s *= ms
            assert 4 % s != 0, f'{format_str.format(i+1, j)}, shape: {model_sd[format_str.format(i+1, j)].shape}'

    for i, num_blocks in enumerate(stage_blocks):
        for j in range(num_blocks):
            write_fp32(model_sd[format_str.format(i+1, j)], file)


def fp32_write_img_backbone(model_sd, cfg, file):
    # conv1 meta
    shape = model_sd['img_backbone.conv1.weight'].shape
    file.write(struct.pack('iiii', shape[0], shape[1], shape[2], shape[3]))
    file.write(struct.pack('iiii', 7, 2, 3, 1))
    assert 4 % sum(shape), f'img_backbone.conv1, shape: {shape}'
    # maxpool meta
    file.write(struct.pack('iiii', 3, 2, 1, 1))

    arch_settings = {
        18: (2, 2, 2, 2),
        34: (3, 4, 6, 3),
        50: (3, 4, 6, 3),
        101: (3, 4, 23, 3),
        152: (3, 8, 36, 3)
    }

    depth = cfg['img_backbone']['depth']
    stage_blocks = arch_settings[depth]
    
    # layer conv1 meta
    for i, num_blocks in enumerate(stage_blocks):
        for j in range(num_blocks):
            shape = model_sd[f'img_backbone.layer{i+1}.{j}.conv1.weight'].shape
            file.write(struct.pack('iiii', shape[0], shape[1], shape[2], shape[3]))
            if i != 0 and j == 0:
                file.write(struct.pack('iiii', 1, 2, 0, 1))
            else:
                file.write(struct.pack('iiii', 1, 1, 0, 1))
    # layer conv2 meta
    for i, num_blocks in enumerate(stage_blocks):
        for j in range(num_blocks):
            shape = model_sd[f'img_backbone.layer{i+1}.{j}.conv2.weight'].shape
            file.write(struct.pack('iiii', shape[0], shape[1], shape[2], shape[3]))
            file.write(struct.pack('iiii', 3, 1, 1, 1))
    # layer conv2 conv_deform meta
    for i, num_blocks in enumerate(stage_blocks):
        for j in range(num_blocks):
            if i >= 2:
                shape = model_sd[f'img_backbone.layer{i+1}.{j}.conv2.conv_offset.weight'].shape
                file.write(struct.pack('iiii', shape[0], shape[1], shape[2], shape[3]))
                file.write(struct.pack('iiii', 3, 1, 1, 1))
    # layer conv3 meta
    for i, num_blocks in enumerate(stage_blocks):
        for j in range(num_blocks):
            shape = model_sd[f'img_backbone.layer{i+1}.{j}.conv3.weight'].shape
            file.write(struct.pack('iiii', shape[0], shape[1], shape[2], shape[3]))
            file.write(struct.pack('iiii', 1, 1, 0, 1))
    # downsample 0 conv meta
    for i, num_blocks in enumerate(stage_blocks):
        shape = model_sd[f'img_backbone.layer{i+1}.0.downsample.0.weight'].shape
        file.write(struct.pack('iiii', shape[0], shape[1], shape[2], shape[3]))
        if i == 0:
            file.write(struct.pack('iiii', 1, 1, 0, 1))
        else:
            file.write(struct.pack('iiii', 1, 2, 0, 1))

    # conv1
    write_fp32(model_sd['img_backbone.conv1.weight'], file)
    # print(f"conv1: {model_sd['img_backbone.conv1.weight']}")
    # bn1
    write_fp32(model_sd['img_backbone.bn1.weight'], file)
    write_fp32(model_sd['img_backbone.bn1.bias'], file)
    write_fp32(model_sd['img_backbone.bn1.running_mean'], file)
    write_fp32(model_sd['img_backbone.bn1.running_var'], file)
    # print(f"bn1: {model_sd['img_backbone.bn1.weight']}, {model_sd['img_backbone.bn1.bias']}, {model_sd['img_backbone.bn1.running_mean']}, {model_sd['img_backbone.bn1.running_var']}")
    # print(f'conv1: {model_sd["img_backbone.conv1.weight"].shape}, {model_sd["img_backbone.bn1.bias"].shape}, {model_sd["img_backbone.bn1.running_mean"].shape}, {model_sd["img_backbone.bn1.running_var"].shape}')

    # layer conv1
    fp32_write_img_backbone_layer(model_sd, 'img_backbone.layer{}.{}.conv1.weight', stage_blocks, file)
    # layer bn1
    fp32_write_img_backbone_layer(model_sd, 'img_backbone.layer{}.{}.bn1.weight', stage_blocks, file)
    fp32_write_img_backbone_layer(model_sd, 'img_backbone.layer{}.{}.bn1.bias', stage_blocks, file)
    fp32_write_img_backbone_layer(model_sd, 'img_backbone.layer{}.{}.bn1.running_mean', stage_blocks, file)
    fp32_write_img_backbone_layer(model_sd, 'img_backbone.layer{}.{}.bn1.running_var', stage_blocks, file)
    # layer conv2
    fp32_write_img_backbone_layer(model_sd, 'img_backbone.layer{}.{}.conv2.weight', stage_blocks, file)
    # layer bn2
    fp32_write_img_backbone_layer(model_sd, 'img_backbone.layer{}.{}.bn2.weight', stage_blocks, file)
    fp32_write_img_backbone_layer(model_sd, 'img_backbone.layer{}.{}.bn2.bias', stage_blocks, file)
    fp32_write_img_backbone_layer(model_sd, 'img_backbone.layer{}.{}.bn2.running_mean', stage_blocks, file)
    fp32_write_img_backbone_layer(model_sd, 'img_backbone.layer{}.{}.bn2.running_var', stage_blocks, file)

    # layer conv2 conv_deform
    for i, num_blocks in enumerate(stage_blocks):
        for j in range(num_blocks):
            if i >=2:
                write_fp32(model_sd[f'img_backbone.layer{i+1}.{j}.conv2.conv_offset.weight'], file)
                print(model_sd[f'img_backbone.layer{i+1}.{j}.conv2.conv_offset.weight'].shape)

    # layer conv2 conv_deform
    for i, num_blocks in enumerate(stage_blocks):
        for j in range(num_blocks):
            if i >=2:
                write_fp32(model_sd[f'img_backbone.layer{i+1}.{j}.conv2.conv_offset.bias'], file)
                print(model_sd[f'img_backbone.layer{i+1}.{j}.conv2.conv_offset.bias'].shape)

    # layer conv3
    fp32_write_img_backbone_layer(model_sd, 'img_backbone.layer{}.{}.conv3.weight', stage_blocks, file)
    # layer bn3
    fp32_write_img_backbone_layer(model_sd, 'img_backbone.layer{}.{}.bn3.weight', stage_blocks, file)
    fp32_write_img_backbone_layer(model_sd, 'img_backbone.layer{}.{}.bn3.bias', stage_blocks, file)
    fp32_write_img_backbone_layer(model_sd, 'img_backbone.layer{}.{}.bn3.running_mean', stage_blocks, file)
    fp32_write_img_backbone_layer(model_sd, 'img_backbone.layer{}.{}.bn3.running_var', stage_blocks, file)

    # downsample 0 conv
    for i, num_blocks in enumerate(stage_blocks):
        write_fp32(model_sd[f'img_backbone.layer{i+1}.0.downsample.0.weight'], file)
    # downsample 0 bn
    for i, num_blocks in enumerate(stage_blocks):
        write_fp32(model_sd[f'img_backbone.layer{i+1}.0.downsample.1.weight'], file)
    for i, num_blocks in enumerate(stage_blocks):
        write_fp32(model_sd[f'img_backbone.layer{i+1}.0.downsample.1.bias'], file)
    for i, num_blocks in enumerate(stage_blocks):
        write_fp32(model_sd[f'img_backbone.layer{i+1}.0.downsample.1.running_mean'], file)
    for i, num_blocks in enumerate(stage_blocks):
        write_fp32(model_sd[f'img_backbone.layer{i+1}.0.downsample.1.running_var'], file)