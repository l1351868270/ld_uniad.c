

import argparse
import math


def arithmetic_intensity(M, N, K):
    # Compute the number of floating-point operations
    ops = 2 * M * N * K
    bytes = 2 * (M * K + N * K + M * N)
    
    ai = ops / bytes
    print(f"gemm: {M}x{N}x{K} arithmetic intensity: {ai:.5f}")
    return ai

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='compute the convolutional arithmetic intensity')
    parser.add_argument('-M', help='', type=int, default=512)
    parser.add_argument("-N", help='', type=int, default=1024)
    parser.add_argument("-K", help='', type=int, default=4096)
    args = parser.parse_args()

    M = args.M
    N = args.N
    K = args.K
    ai = arithmetic_intensity(M, N, K)
