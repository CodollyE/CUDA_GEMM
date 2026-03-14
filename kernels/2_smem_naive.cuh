template<typename T, const int BLOCK_SIZE>
__global__ void smem_naive_kernel(int M, int N, int K, T alpha, T *A, T *B, T beta, T *C){
  __shared__ T As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ T Bs[BLOCK_SIZE][BLOCK_SIZE];
  
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  T tmp = T(0);
  for(int offset = 0; offset < K; offset += BLOCK_SIZE){
    As[threadIdx.y][threadIdx.x] = row * K + offset + threadIdx.x < M * K ? A[row * K + offset + threadIdx.x] : T(0);
    Bs[threadIdx.y][threadIdx.x] = (offset + threadIdx.y) * N + col < K * N ? B[(offset + threadIdx.y) * N + col] : T(0);
    __syncthreads();

    for(int k = 0; k < BLOCK_SIZE; k++){
        tmp += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    __syncthreads();
  }

  if(row < M && col < N){
    C[row * N + col] = alpha * tmp + beta * C[row * N + col];
  }
  
}