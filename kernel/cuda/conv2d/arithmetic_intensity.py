

import argparse
import math

def conv2d_arithmetic_intensity(N, C, H, W, K, R, S, U, V, PadH, PadW, DilH, DilW):
    # https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html
    # Compute the output spatial dimensions
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    P = math.floor((H + 2 * PadH - DilH * (R - 1) - 1) / U + 1)
    Q = math.floor((W + 2 * PadW - DilW * (S - 1) - 1) / V + 1)

    # Compute the number of floating-point operations
    v_M = N * P * Q
    v_N = K
    v_K = C * R * S

    ops = 2 * v_M * v_N * v_K
    bytes = 2 * (N * C * H * W + K * C * R * S + N * K * P * Q)
    
    ai = ops / bytes
    print(f"Conv2d: {N}x{C}x{H}x{W} conv {K}x{C}x{R}x{S} -> {N}x{K}x{P}x{Q}, arithmetic intensity: {ai:.5f}")
    return ai

def im2col_arithmetic_intensity(N, C, H, W, K, R, S, U, V, PadH, PadW, DilH, DilW):
    # https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
    # Compute the output spatial dimensions
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    P = math.floor((H + 2 * PadH - DilH * (R - 1) - 1) / U + 1)
    Q = math.floor((W + 2 * PadW - DilW * (S - 1) - 1) / V + 1)

    # Compute the number of floating-point operations
    v_M = N * P * Q
    v_N = K
    v_K = C * R * S

    ops = 2 * v_M * v_N * v_K
    bytes = 2 * (v_M * v_K + v_N * v_K + v_M * v_N)
    
    ai = ops / bytes
    print(f"im2col Conv2d: {N}x{C}x{H}x{W} conv {K}x{C}x{R}x{S} -> {N}x{K}x{P}x{Q}, {v_M}x{v_N}x{v_K} arithmetic intensity: {ai:.5f}")
    return ai

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='compute the convolutional arithmetic intensity')
    parser.add_argument('-N', help='Batch size', type=int, default=256)
    parser.add_argument('-C', help='Input: Number of channels', type=int, default=64)
    parser.add_argument("-H", help='Input: Height', type=int, default=928)
    parser.add_argument("-W", help='Input: Width', type=int, default=1600)
    
    parser.add_argument("-K", help='Output: Number of channels', type=int, default=64)
    parser.add_argument("-R", help='Filter: Height', type=int, default=3)
    parser.add_argument("-S", help='Filter: Width', type=int, default=3)

    parser.add_argument("-U", help='Filter: Vertical stride', type=int, default=1)
    parser.add_argument("-V", help='Filter: Horizontal strid', type=int, default=1)
    parser.add_argument("-PadH", help='Filter: Input padding in the vertical dimension', type=int, default=0)
    parser.add_argument("-PadW", help='Filter: Input padding in the horizontal dimension', type=int, default=0)
    parser.add_argument("-DilH", help='Filter: Dilation in the vertical dimension', type=int, default=1)
    parser.add_argument("-DilW", help='Filter: Dilation in the horizontal dimension', type=int, default=1)

    args = parser.parse_args()

    N = args.N
    C = args.C
    H = args.H
    W = args.W

    K = args.K
    R = args.R
    S = args.S
    U = args.U
    V = args.V
    PadH = args.PadH
    PadW = args.PadW
    DilH = args.DilH
    DilW = args.DilW
    # N = 1
    ai = conv2d_arithmetic_intensity(N, C, H, W, K, R, S, U, V, PadH, PadW, DilH, DilW)
    ai = im2col_arithmetic_intensity(N, C, H, W, K, R, S, U, V, PadH, PadW, DilH, DilW)


    N = 1; C = 3; H = 928; W = 1600; K = 64; R = 7; S = 7; U = 2; V = 2; PadH = 3; PadW = 3; DilH = 1; DilW = 1
    P = math.floor((H + 2 * PadH - DilH * (R - 1) - 1) / U + 1)
    Q = math.floor((W + 2 * PadW - DilW * (S - 1) - 1) / V + 1)
    ai = conv2d_arithmetic_intensity(N, C, H, W, K, R, S, U, V, PadH, PadW, DilH, DilW)
    ai = im2col_arithmetic_intensity(N, C, H, W, K, R, S, U, V, PadH, PadW, DilH, DilW)

    P /= 2
    Q /= 2

    N = 1; C = 64; H = P; W = Q; K = 64; R = 1; S = 1; U = 1; V = 1; PadH = 0; PadW = 0; DilH = 1; DilW = 1
    P = math.floor((H + 2 * PadH - DilH * (R - 1) - 1) / U + 1)
    Q = math.floor((W + 2 * PadW - DilW * (S - 1) - 1) / V + 1)
    ai = conv2d_arithmetic_intensity(N, C, H, W, K, R, S, U, V, PadH, PadW, DilH, DilW)
    ai = im2col_arithmetic_intensity(N, C, H, W, K, R, S, U, V, PadH, PadW, DilH, DilW)
    N = 1; C = 64; H = P; W = Q; K = 64; R = 3; S = 3; U = 1; V = 1; PadH = 1; PadW = 1; DilH = 1; DilW = 1
    P = math.floor((H + 2 * PadH - DilH * (R - 1) - 1) / U + 1)
    Q = math.floor((W + 2 * PadW - DilW * (S - 1) - 1) / V + 1)
    ai = conv2d_arithmetic_intensity(N, C, H, W, K, R, S, U, V, PadH, PadW, DilH, DilW)
    ai = im2col_arithmetic_intensity(N, C, H, W, K, R, S, U, V, PadH, PadW, DilH, DilW)
    N = 1; C = 64; H = P; W = Q; K = 256; R = 1; S = 1; U = 1; V = 1; PadH = 0; PadW = 0; DilH = 1; DilW = 1
    ai = conv2d_arithmetic_intensity(N, C, H, W, K, R, S, U, V, PadH, PadW, DilH, DilW)
    ai = im2col_arithmetic_intensity(N, C, H, W, K, R, S, U, V, PadH, PadW, DilH, DilW)