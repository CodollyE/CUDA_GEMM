

// naive implementation of gemm
template<typename T>
__global__ void naive_kernel(int M, int N, int K, T alpha, T *A,
                                  T *B, T beta, T *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 计算当前线程对应的行索引
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 计算当前线程对应的列索引

    if (row < M && col < N) { // 确保线程在矩阵范围内
        float value = 0.0;
        for (int k = 0; k < K; ++k) {
            value += float(A[row * K + k]) * float(B[k * N + col]); // 累加乘积
        }
        C[row * N + col] = T(float(alpha) * value + float(beta) * float(C[row * N + col])); // 更新C矩阵
    }
}