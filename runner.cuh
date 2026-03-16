#pragma once
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cuComplex.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <type_traits>
#include <sys/time.h>
#include <cuda_runtime.h>

#include "kernels/1_naive.cuh"
#include "kernels/2_smem_naive.cuh"
#include "kernels/3_smem_1D_tiling.cuh"

#define cudaCheck(call) cuda_check((call), __FILE__, __LINE__)

inline void cuda_check(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("CUDA error at %s:%d: %s\n", file, line,
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

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
bool verify_matrix(T *matRef, T *matOut, int N) {
  double diff = 0.0;
  int i;
  for (i = 0; i < N; i++) {
    diff = std::fabs(double(matRef[i]) - double(matOut[i]));
    if (isnan(diff) || diff > 0.01) {
      printf("Divergence! Should %5.2f, but now is %5.2f (Diff %5.2f) at %d\n",
             float(matRef[i]), float(matOut[i]), diff, i);
      return false;
    }
  }
  return true;
}


template<typename T>
void print_matrix(const T *A, int M, int N, std::ofstream &fs) {
  int i;
  fs << std::setprecision(2)
     << std::fixed; // Set floating-point precision and fixed notation
  fs << "[";
  for (i = 0; i < M * N; i++) {
    if ((i + 1) % N == 0)
      fs << std::setw(5) << float(A[i]); // Set field width and write the value
    else
      fs << std::setw(5) << float(A[i]) << ", ";
    if ((i + 1) % N == 0) {
      if (i + 1 < M * N)
        fs << ";\n";
    }
  }
  fs << "]\n";
}


template<typename T>
void run_cublas(int M, int N, int K, T alpha, T *A, T *B, T beta,
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


template<typename T>
void run_naive(int M, int N, int K, T alpha, T *A, T *B,
                     T beta, T *C) {
  dim3 block(32, 32);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  naive_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}


template<typename T>
void run_smem_naive(int M, int N, int K, T alpha, T *A, T *B,
                     T beta, T *C) {
  constexpr size_t smem_dim = 32;
  dim3 block(32, 32);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

  // 让运行时在启动该核时优先分配更多的共享内存、牺牲 L1 缓存，适合大量使用 SMEM、而不依赖 L1 的内核
  cudaCheck(cudaFuncSetAttribute(smem_naive_kernel<T, smem_dim>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared));
  smem_naive_kernel<T, smem_dim><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}


template<typename T>
void run_smem_1D_tiling(int M, int N, int K, T alpha, T *A, T *B,
                     T beta, T *C) {
  // size of tile
  constexpr size_t TILE_M = 64;
  constexpr size_t TILE_K = 8;
  constexpr size_t TILE_N = 64;
  // size of item per thread
  constexpr size_t THREAD_M = 8;

  dim3 block((TILE_M * TILE_N) / THREAD_M);
  dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
  smem_1D_tiling<T, TILE_M, TILE_N, TILE_K, THREAD_M><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}


template<typename T>
void run_kernel(int kernel_num, int M, int N, int K, T alpha, T *A,
                T *B, T beta, T *C, cublasHandle_t handle){
  switch (kernel_num) {
    case 0:
      run_cublas(M, N, K, alpha, A, B, beta, C, handle);
      break;
    case 1:
      run_naive(M, N, K, alpha, A, B, beta, C);
      break;
    case 2:
      run_smem_naive(M, N, K, alpha, A, B, beta, C);
      break;
    case 3:
      run_smem_1D_tiling(M, N, K, alpha, A, B, beta, C);
      break;
    }
}


template<typename T>
void runner(int kernel_num){
  const std::string errLogFile = "matrixValidationFailure.txt"; // log file
  
  // Declare the handle, create the handle, cublasCreate will return a value of
  // type cublasStatus_t to determine whether the handle was created
  // successfully (the value is 0)
  cublasHandle_t handle;
  if (cublasCreate(&handle)) {
    std::cerr << "Create cublas handle error." << std::endl;
    exit(EXIT_FAILURE);
  };

  // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
  // publishing event tasks in the target stream
  float elapsed_time, cublas_elapsed_time;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  // cuBLAS FLOPs ceiling is reached at 8192
  std::vector<int> SIZE = {128, 256, 512, 1024, 2048, 4096};

  long m, n, k, max_size;
  max_size = SIZE[SIZE.size() - 1];
  std::cout << "Max size: " << max_size << std::endl;

  // Host and device pointers (opaque, choose element size and cuBLAS types at runtime)
  T *A = nullptr, *B = nullptr, *C = nullptr, *C_ref = nullptr; // host
  T *dA = nullptr, *dB = nullptr, *dC = nullptr, *dC_ref = nullptr; // device

  size_t type_size = sizeof(T);

  // allocate host and device buffers using the chosen type_size
  size_t total_elems = static_cast<size_t>(max_size) * static_cast<size_t>(max_size);
  size_t total_bytes = type_size * total_elems;

  // host allocations
  A = static_cast<T *>(malloc(total_bytes));
  B = static_cast<T *>(malloc(total_bytes));
  C = static_cast<T *>(malloc(total_bytes));
  C_ref = static_cast<T *>(malloc(total_bytes));

  // device allocations
  cudaCheck(cudaMalloc(&dA, total_bytes));
  cudaCheck(cudaMalloc(&dB, total_bytes));
  cudaCheck(cudaMalloc(&dC, total_bytes));
  cudaCheck(cudaMalloc(&dC_ref, total_bytes));
  
  // randomize host matrices
  randomize_matrix(A, total_elems);
  randomize_matrix(B, total_elems);
  randomize_matrix(C, total_elems);
  memset(C, 0, total_bytes); // Set C to zero for better numerical stability in verification

  // set alpha and beta use A[0] and B[0]
  T alpha = A[0], beta = B[0]; // GEMM input parameters, C=α*AB+β*C

  // copy host -> device
  cudaCheck(cudaMemcpy(dA, A, total_bytes, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dB, B, total_bytes, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC, C, total_bytes, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC_ref, C, total_bytes, cudaMemcpyHostToDevice));

  int repeat_times = 50;
  for (int size : SIZE) {
    m = n = k = size;

    std::cout << "dimensions(m=n=k) " << m << ", alpha(as float): " << float(alpha)
              << ", beta(as float): " << float(beta) << std::endl;
    // Verify the correctness of the calculation, and execute it once before the
    // kernel function timing to avoid cold start errors
    if (kernel_num != 0) {
      run_kernel(0, m, n, k, alpha, dA, dB, beta, dC_ref,
                 handle); // cuBLAS
      run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC,
                 handle); // Executes the kernel, modifies the result matrix
      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
      cudaMemcpy(C, dC, total_bytes, cudaMemcpyDeviceToHost);
      cudaMemcpy(C_ref, dC_ref, total_bytes, cudaMemcpyDeviceToHost);

      if (!verify_matrix(C_ref, C, m * n)) {
        std::cout
            << "Failed to pass the correctness verification against NVIDIA "
               "cuBLAS."
            << std::endl;
        if (m <= 128) {
          std::cout << " Logging faulty output into " << errLogFile << "\n";
          std::ofstream fs;
          fs.open(errLogFile);
          fs << "A:\n";
          print_matrix(A, m, n, fs);
          fs << "B:\n";
          print_matrix(B, m, n, fs);
          fs << "C:\n";
          print_matrix(C, m, n, fs);
          fs << "Should:\n";
          print_matrix(C_ref, m, n, fs);
        }
        exit(EXIT_FAILURE);
      }
    }

    // get cublas FLOPs ceiling for reference
    cudaEventRecord(beg);
    for (int j = 0; j < repeat_times; j++) {
      // We don't reset dC between runs to save time
      run_kernel(0, m, n, k, alpha, dA, dB, beta, dC_ref, handle);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&cublas_elapsed_time, beg, end);
    cublas_elapsed_time /= 1000.; // Convert to seconds

    // get kernel FLOPs and execution time
    cudaEventRecord(beg);
    for (int j = 0; j < repeat_times; j++) {
      // We don't reset dC between runs to save time
      run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.; // Convert to seconds

    long flops = 2 * m * n * k;
    printf(
        "Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS, (%7.2f%%) of cublas . size: "
        "(%ld).\n",
        elapsed_time / repeat_times,
        (repeat_times * flops * 1e-9) / elapsed_time,
        ((repeat_times * flops * 1e-9) / elapsed_time) / ((repeat_times * flops * 1e-9) / cublas_elapsed_time) * 100, m);
    fflush(stdout);
    // make dC and dC_ref equal again (all set 0)
    cudaCheck(cudaMemset(dC, 0, total_bytes));
    cudaCheck(cudaMemset(dC_ref, 0, total_bytes));
  }

  // Free up CPU and GPU space
  free(A);
  free(B);
  free(C);
  free(C_ref);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dC_ref);
  cublasDestroy(handle);

}
