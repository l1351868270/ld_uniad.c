

MODEL_LIANMENT = 16

def write_fp32(tensor, file):
    file.write(tensor.detach().cpu().numpy().astype("float32").tobytes())