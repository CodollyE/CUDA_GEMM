// #include "kernel.cuh"
#include "runner.cuh"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <cuComplex.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <type_traits>

// 类型转换映射： c++ -> cudaDataType
template<typename T> struct cudaDataTypeMap;
// 实数类型
template<>
struct cudaDataTypeMap<__half> {
  static constexpr cudaDataType_t type = CUDA_R_16F;
  static constexpr cublasComputeType_t compute = CUBLAS_COMPUTE_32F_FAST_16F;
};

template<>
struct cudaDataTypeMap<__nv_bfloat16> {
  static constexpr cudaDataType_t type = CUDA_R_16BF;
  static constexpr cublasComputeType_t compute = CUBLAS_COMPUTE_32F_FAST_16BF;
};

template<>
struct cudaDataTypeMap<float> {
  static constexpr cudaDataType_t type = CUDA_R_32F;
  static constexpr cublasComputeType_t compute = CUBLAS_COMPUTE_32F;
};

template<>
struct cudaDataTypeMap<double> {
  static constexpr cudaDataType_t type = CUDA_R_64F;
  static constexpr cublasComputeType_t compute = CUBLAS_COMPUTE_64F;
};

template<>
struct cudaDataTypeMap<int8_t> {
  static constexpr cudaDataType_t type = CUDA_R_8I;
  static constexpr cublasComputeType_t compute = CUBLAS_COMPUTE_32I;
};

template<>
struct cudaDataTypeMap<uint8_t> {
  static constexpr cudaDataType_t type = CUDA_R_8U;
  static constexpr cublasComputeType_t compute = CUBLAS_COMPUTE_32I;
};

template<>
struct cudaDataTypeMap<int32_t> {
  static constexpr cudaDataType_t type = CUDA_R_32I;
  static constexpr cublasComputeType_t compute = CUBLAS_COMPUTE_32I;
};

template<>
struct cudaDataTypeMap<uint32_t> {
  static constexpr cudaDataType_t type = CUDA_R_32U;
  static constexpr cublasComputeType_t compute = CUBLAS_COMPUTE_32I;
};

//pair类型
template<>
struct cudaDataTypeMap<__half2> {
  static constexpr cudaDataType_t type = CUDA_C_16F;
  static constexpr cublasComputeType_t compute = CUBLAS_COMPUTE_32F_FAST_16F;
};

template<>
struct cudaDataTypeMap<__nv_bfloat162> {
  static constexpr cudaDataType_t type = CUDA_C_16BF;
  static constexpr cublasComputeType_t compute = CUBLAS_COMPUTE_32F_FAST_16BF;
};
//虚数类型
template<>
struct cudaDataTypeMap<cuComplex> {
  static constexpr cudaDataType_t type = CUDA_C_32F;
  static constexpr cublasComputeType_t compute = CUBLAS_COMPUTE_32F;
};

template<>
struct cudaDataTypeMap<cuDoubleComplex> {
  static constexpr cudaDataType_t type = CUDA_C_64F;
  static constexpr cublasComputeType_t compute = CUBLAS_COMPUTE_64F;
};



void cudaCheck(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("CUDA error at %s:%d: %s\n", file, line,
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
    else if constexpr (std::is_same_v<T, __nv_bfloat162>) {
      float imag = (rand() % 5) + 0.01f * (rand() % 5);
      imag = (rand() % 2 == 0) ? imag : -imag;
      __nv_bfloat162 v;
      v.x = __float2bfloat16(real);
      v.y = __float2bfloat16(imag);
      mat[i] = v;
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
  // cublas默认认为A矩阵是经过转置的（即列主序），比如1，2，3，4，5，6的矩阵在cublas中会被认为是[[1,4],[2,5],[3,6]]
  // 但注意这只是逻辑上的认为，内存里依旧没有变
  // 因此我们需要把B放在前面，A放在后面，并且N,M对换，这样才是B^T * A^T，
  // 因为我们用CUBLAS_OP_N说明不需要再转置了（因为cublas无论如何都认为是列主序，如果要转置那他就转置乘行主序）
  // 如果用CUBLAS_OP_T表示要转置，这时候才是A在前B在后，N,M不对换
  constexpr cudaDataType_t data_type = cudaDataTypeMap<T>::type;
  constexpr auto compute_type = cudaDataTypeMap<T>::compute;
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, 
                B, data_type, N, A, data_type, K, &beta, C, data_type, N, 
                compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}


void run_kernel(int kernel_num, int M, int N, int K, float alpha, float *A,
                float *B, float beta, float *C, cublasHandle_t handle){
    switch (kernel_num) {
        case 0:
            run_cublas_gemm(M, N, K, alpha, A, B, beta, C, handle);
            break;
        // case 1:
        //     run_naive_gemm();
        //     break;
        // case 2:
        //     run_shared_gemm();
        //     break;
    }
}