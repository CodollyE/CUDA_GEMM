#include "kernel.cuh"
#include "runner.cuh"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <cuComplex.h>

// 类型转换映射： c++ -> cudaDataType
<typename T> struct cudaDataType;
// 实数类型
template<> struct cudaDataType<half> {
  static constexpr cudaDataType_t type = CUDA_R_16F;
};
template<> struct cudaDataType<__half> {
  static constexpr cudaDataType_t type = CUDA_R_16F;
};
template<> struct cudaDataType<__nv__bfloat16> {
  static constexpr  cudaDataType_t type = CUDA_R_16BF;
};
template<> struct cudaDataType<float> {
  static constexpr  cudaDataType_t type = CUDA_R_32F;
};
template<> struct cudaDataType<double> {
  static constexpr  cudaDataType_t type = CUDA_R_64F;
};
template<> struct cudaDataType<int8_t> {
  static constexpr  cudaDataType_t type = CUDA_R_8I;
};
template<> struct cudaDataType<uint8_t> {
  static constexpr  cudaDataType_t type = CUDA_R_8U;
};
template<> struct cudaDataType<int32_t> {
  static constexpr  cudaDataType_t type = CUDA_R_32I;
};
template<> struct cudaDataType<uint32_t> {
  static constexpr  cudaDataType_t type = CUDA_R_32U;
};
//pair类型
template<> struct cudaDataType<__half2> {
  static constexpr  cudaDataType_t type = CUDA_C_16F;
};
template<> struct cudaDataType<__nv__bfloat162> {   
  static constexpr  cudaDataType_t type = CUDA_C_16BF;
};
//虚数类型
template<> struct cudaDataType<cuComplex> {
  static constexpr  cudaDataType_t type = CUDA_C_32F;
};
template<> struct cudaDataType<cuDoubleComplex> {
  static constexpr  cudaDataType_t type = CUDA_C_64F;
};



void cudaCheck(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf(stderr, "CUDA error at %s:%d: %s\n", file, line,
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

void CudaDeviceInfo() {
  int deviceId;

  cudaGetDevice(&deviceId);

  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, deviceId);

  printf("Device ID: %d\n\
    Name: %s\n\
    Compute Capability: %d.%d\n\
    memoryBusWidth: %d\n\
    maxThreadsPerBlock: %d\n\
    maxThreadsPerMultiProcessor: %d\n\
    maxRegsPerBlock: %d\n\
    maxRegsPerMultiProcessor: %d\n\
    totalGlobalMem: %zuMB\n\
    sharedMemPerBlock: %zuKB\n\
    sharedMemPerMultiprocessor: %zuKB\n\
    totalConstMem: %zuKB\n\
    multiProcessorCount: %d\n\
    Warp Size: %d\n",
         deviceId, props.name, props.major, props.minor, props.memoryBusWidth,
         props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor,
         props.regsPerBlock, props.regsPerMultiprocessor,
         props.totalGlobalMem / 1024 / 1024, props.sharedMemPerBlock / 1024,
         props.sharedMemPerMultiprocessor / 1024, props.totalConstMem / 1024,
         props.multiProcessorCount, props.warpSize);
};

template<typename T>
void randomize_matrix(T *mat, int N) {
  // NOTICE: Use gettimeofday instead of srand((unsigned)time(NULL)); the time
  // precision is too low and the same random number is generated.
  struct timeval time {};
  gettimeofday(&time, nullptr);
  srand(time.tv_usec);
  for (int i = 0; i < N; i++) {

    float real = (rand() % 5) + 0.01f * (rand() % 5);
    real = (rand() % 2 == 0) ? real : -real;
    if constexpr (std::is_same_v<T, __half2>) {
      float imag = (rand() % 5) + 0.01f * (rand() % 5);
      imag = (rand() % 2 == 0) ? imag : -imag;
      mat[i] = make_half2(__float2half(real),__float2half(imag));
    }
    else if constexpr (std::is_same_v<T, __nv__bfloat162>) {
      float imag = (rand() % 5) + 0.01f * (rand() % 5);
      imag = (rand() % 2 == 0) ? imag : -imag;
      mat[i] = make_nv_bfloat162(__float2bfloat16(real), __float2bfloat16(imag));
    }
    else if constexpr (std::is_same_v<T, cuComplex>) {
      float imag = (rand() % 5) + 0.01f * (rand() % 5);
      imag = (rand() % 2 == 0) ? imag : -imag;
      mat[i] = make_cuComplex(real, imag);
    }
    else if constexpr (std::is_same_v<T, cuDoubleComplex>) {
      double imag = (rand() % 5) + 0.01 * (rand() % 5);
      imag = (rand() % 2 == 0) ? imag : -imag;
      mat[i] = make_cuDoubleComplex(real, imag);
    } 
    else {
      mat[i] = static_cast<T>(real);
    }
  }
}

template<typename T>
void run_cublas_gemm(int M, int N, int K, float alpha, T *A, T *B, float beta,
                 T *C, cublasHandle_t handle) {
  // cublasSgemm: C = alpha * op(A) * op(B) + beta * C
  // cuBLAS uses column-major order. So we change the order of our row-major A & B
  // because (B^T * A^T)^T = A * B
  // op()
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, T,
              N, A, T, K, &beta, C, T, N, CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void run_kernel(int kernel_num, int M, int N, int K, float alpha, float *A,
                float *B, float beta, float *C, cublasHandle_t handle){
    switch (kernel_num) {
        case 0:
            run_cublas_gemm();
            break;
        case 1:
            run_naive_gemm();
            break;
        case 2:
            run_shared_gemm();
            break;
    }
}