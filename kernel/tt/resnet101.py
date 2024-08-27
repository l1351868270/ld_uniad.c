import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
from calflops import calculate_flops
from torchprofile import profile_macs

model = models.resnet101(pretrained=False)

dummy_input = torch.randn(1, 3, 928, 1600)

with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
    with record_function("model_inference"):
        model(dummy_input)

print(prof.key_averages().table(sort_by="flops", row_limit=10))


input_shape = (1, 3, 928, 1600)
flops, macs, params = calculate_flops(model=model, 
                                      input_shape=input_shape,
                                      output_as_string=True,
                                      output_precision=4)
print("Alexnet FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))


macs = profile_macs(model, dummy_input)
flops = 2 * macs  # Since 1 MAC (Multiply-Accumulate Operation) is 2 FLOPs
print(f"FLOPs: {flops}")