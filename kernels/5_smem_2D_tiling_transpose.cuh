template<typename T, const size_t TILE_M, const size_t TILE_N, const size_t TILE_K, const size_t THREAD_M, const size_t THREAD_N>
__global__ void smem_2D_tiling_transpose(int M, int N, int K, T alpha, T *A, T *B, T beta, T *C){
  int thread_num = (TILE_M * TILE_N) / (THREAD_M * THREAD_N);
  int strideA = thread_num / TILE_K;
  int strideB = thread_num / TILE_N;

  __shared__ T As[TILE_M * TILE_K];
  __shared__ T Bs[TILE_K * TILE_N];

  int row = blockIdx.y * TILE_M;
  int col = blockIdx.x * TILE_N;
  int thread_per_row = TILE_N / THREAD_N;
  int thread_row = threadIdx.x / thread_per_row;
  int thread_col = threadIdx.x % thread_per_row;

  A += row * K;
  B += col;
  C += row * N + col; 

  T reg_A[THREAD_M] = {0};
  T reg_B[THREAD_N] = {0};
  T thread_result[THREAD_M * THREAD_N] = {0};
  
  for(int bk = 0; bk < K; bk += TILE_K){
    // move data to smem
    // 因为总共只有 (TILE_M * TILE_N) / (THREAD_M * THREAD_N) 个线程, 而填满As和Bs需要 TILE_M * TILE_K 和 TILE_K * TILE_N 个元素
    // 因此填满As需要每个线程填 TILE_K * THREAD_M * THREAD_N / TILE_N 个元素，因此一个线程要填多个元素，用循环解决
    // 我们可以用滑动窗口来解决这个问题，我们把线程看作一个二维矩阵，这个二维矩阵中所有线程都能一口气把数据都读进共享内存里
    // 以As为例，这个数据区域行数：TILE_M, 行数：strideA = 线程数 / TILE_M. 然后这个区域一直往下滑，直到把TILE_M * TILE_K填满
    // 同理可以换一个方向滑动 (A、B分别沿着TILE_M和TILE_K滑动，这样可以内存连续)
    // for(int offset = 0; offset < TILE_K; offset += strideA){
    //     As[(threadIdx.x / strideA) * TILE_K + offset + (threadIdx.x % strideA)] = A[(threadIdx.x / strideA) * K + offset + (threadIdx.x % strideA)];
    // }
    for(int offset = 0; offset < TILE_M; offset += strideA){
        As[(threadIdx.x % TILE_K) * TILE_M + offset + threadIdx.x / TILE_K] = A[(offset + threadIdx.x / TILE_K) * K + threadIdx.x % TILE_K];
    }
    for(int offset = 0; offset < TILE_K; offset += strideB){
        Bs[(offset + (threadIdx.x / TILE_N)) * TILE_N + (threadIdx.x % TILE_N)] = B[(offset + (threadIdx.x / TILE_N)) * N + (threadIdx.x % TILE_N)];
    }
    __syncthreads();

    for(int k = 0; k < TILE_K; k++){
        for(int i = 0; i < THREAD_M; i++){
            reg_A[i] = As[k * TILE_M + thread_row * THREAD_M + i];
        }
        for(int i = 0; i < THREAD_N; i++){
            reg_B[i] = Bs[k * TILE_N + thread_col * THREAD_N + i];
        }

        for(int i = 0; i < THREAD_M; i++){
            for(int j = 0; j < THREAD_N; j++){
                thread_result[i * THREAD_N + j] += reg_A[i] * reg_B[j];
            }
        }
    }
    __syncthreads();

    A += TILE_K;
    B += TILE_K * N;
  }

  for(int i = 0; i < THREAD_M; i++){
    for(int j = 0; j < THREAD_N; j++){
      C[(thread_row * THREAD_M + i) * N + thread_col * THREAD_N + j] = alpha * thread_result[i * THREAD_N + j] + beta * C[(thread_row * THREAD_M + i) * N + thread_col * THREAD_N + j];
    }
  }
}