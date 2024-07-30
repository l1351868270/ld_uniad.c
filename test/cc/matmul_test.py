import torch
import numpy as np
import time

def test_matmul():
    M = 2227200
    N = 64
    K = 147
    begin = time.time()
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.matmul(A, B)
    end = time.time()
    print(f"np.matmul time: {end-begin:.5f}")
    A = torch.tensor(A)
    B = torch.tensor(B)
    begin = time.time()
    C = torch.matmul(A, B)
    end = time.time()
    print(f"pytorch.matmul time: {end-begin:.5f}")

    N = 64
    C = 3
    H = 928
    W = 1600
    
    C_out = 64
    ksize = 7
    stride = 2
    padding = 3
    dialation = 1

    torch.im2col
    input = np.random.rand(N, C, H, W).astype(np.float32)
    input = torch.tensor(input)
    conv = torch.nn.Conv2d(C, C_out, 7, stride=stride, padding=padding)
    conv.weight = torch.nn.Parameter(torch.rand(C_out, C, ksize, ksize))
    begin = time.time()
    conv(input)
    end = time.time()
    print(f"pytorch conv time: {end-begin:.5f}")

if __name__ == "__main__":
    test_matmul()
