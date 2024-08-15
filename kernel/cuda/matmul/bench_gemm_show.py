import matplotlib.pyplot as plt


if __name__ == "__main__":
    fig, ax = plt.subplots()
    with open("build/bench_gemm.csv", "r") as f:
        lines = f.readline()
        cublas_gemm_gflops = []
        cutlass_gemm_v2_gflops = []
        matrix_size = []
        for line in f:
            print(line)
            elements = line.split(",")
            M = int(elements[0])
            N = int(elements[1])
            K = int(elements[2])
            matrix_size.append(M)
            cublas_gemm_gflops.append(float(elements[3]))
            cutlass_gemm_v2_gflops.append(float(elements[4]))
        # print(lines)
    ax.plot(matrix_size, cublas_gemm_gflops, 'o-', label='cublas_gemm')
    ax.plot(matrix_size, cutlass_gemm_v2_gflops, 'x-', label='cutlass_gemm_v2')
    # ax.plot(matrix_size, blas_gflops, 'g-x', label='blas.matmul')
    # ax.set_xticks(matrix_size[::1])
    ax.tick_params(axis='x', labelrotation=90)
    fig.tight_layout()
    ax.set(xlabel='MNK', ylabel='Performance GFLOPS/sec.', title="Matmul Benchmark")
    ax.grid()
    ax.legend()
    
    # ax.set_xlim([old_data[0,0], old_data[-1,0]])
    # ax.set_ylim([0, max_gflops])
    # plt.xticks()
    fig.savefig("test.png")
    plt.show()