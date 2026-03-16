template<typename T, const size_t TILE_M, const size_t TILE_N, const size_t TILE_K, const size_t THREAD_M>
__global__ void smem_1D_tiling(int M, int N, int K, T alpha, T *A, T *B, T beta, T *C){
  assert(TILE_M * TILE_K == blockDim.x);
  assert(TILE_N * TILE_K == blockDim.x);
  // assert(TILE_M % THREAD_M == 0);

  __shared__ T As[TILE_M * TILE_K];
  __shared__ T Bs[TILE_K * TILE_N];

  int row = blockIdx.y * TILE_M;
  int col = blockIdx.x * TILE_N;

  A += row * K;
  B += col;
  C += row * N + col; 

  T thread_sum[THREAD_M] = {0};
  for(int bk = 0; bk < K; bk += TILE_K){
    int arow = threadIdx.x / TILE_K;
    int acol = threadIdx.x % TILE_K;
    int brow = threadIdx.x / TILE_N;
    int bcol = threadIdx.x % TILE_N;

    As[arow * TILE_K + acol] = (row + arow < M && bk + acol < K) ? A[arow * K + acol] : T(0);
    Bs[brow * TILE_N + bcol] = (bk + brow < K && col + bcol < N) ? B[brow * N + bcol] : T(0);
    __syncthreads();

    for(int k = 0; k < TILE_K; k++){
      T btmp = Bs[k * TILE_N + bcol];
      for(int i = 0; i < THREAD_M; i++){
        thread_sum[i] += As[(threadIdx.x / TILE_N * THREAD_M + i) * TILE_K + k] * btmp;
      }
    }
    __syncthreads();
    
    A += TILE_K;
    B += TILE_K * N;
  }

  for(int i = 0; i < THREAD_M; i++){
    if(row + threadIdx.x / TILE_N * THREAD_M + i < M && col + threadIdx.x % TILE_N < N){
        C[(threadIdx.x / TILE_N * THREAD_M + i) * N + threadIdx.x % TILE_N] = alpha * thread_sum[i] + beta * C[(threadIdx.x / TILE_N * THREAD_M + i) * N + threadIdx.x % TILE_N];
    }
  }
}