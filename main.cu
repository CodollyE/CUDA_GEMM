#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include "runner.cuh"
#include <vector>


#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

const std::string errLogFile = "matrixValidationFailure.txt";

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Please select a kernel (range 0 - 12, 0 for NVIDIA cuBLAS) and provide the data type"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // get kernel number
  int kernel_num = std::stoi(argv[1]);
  if (kernel_num < 0 || kernel_num > 12) {
    std::cerr << "Please enter a valid kernel number (0-12)" << std::endl;
    exit(EXIT_FAILURE);
  }

  // get data type
  std::string data_type = argv[2];
  if (data_type != "float" && data_type != "double") {
    std::cerr << "Please enter a valid data type (float or double)" << std::endl;
    exit(EXIT_FAILURE);
  }

  // get environment variable for device
  int deviceIdx = 0; // gpu id, device指gpu
  if (getenv("DEVICE") != NULL) {
    deviceIdx = atoi(getenv("DEVICE"));
  }
  cudaCheck(cudaSetDevice(deviceIdx));

  printf("Running kernel %d on device %d.\n", kernel_num, deviceIdx);

  // print some device info
  // CudaDeviceInfo();

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
  float elapsed_time;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  // cuBLAS FLOPs ceiling is reached at 8192
  std::vector<int> SIZE = {128, 256, 512, 1024, 2048, 4096};

  long m, n, k, max_size;
  max_size = SIZE[SIZE.size() - 1];
  std::cout << "Max size: " << max_size << std::endl;

  float alpha = 0.5, beta = 3.0; // GEMM input parameters, C=α*AB+β*C

  // Host and device pointers (opaque, choose element size and cuBLAS types at runtime)
  void *A_raw = nullptr, *B_raw = nullptr, *C_raw = nullptr, *C_ref_raw = nullptr; // host
  void *dA_raw = nullptr, *dB_raw = nullptr, *dC_raw = nullptr, *dC_ref_raw = nullptr; // device

  size_t type_size = 0;

  if (data_type == "float") {
    type_size = sizeof(float);
  } else if (data_type == "double") {
    type_size = sizeof(double);
  } else if (data_type == "__half") {
    type_size = sizeof(__half);
  } else if (data_type == "__nv_bfloat16") {
    type_size = sizeof(__nv_bfloat16);
  } else if (data_type == "int8_t") {
    type_size = sizeof(int8_t);
  } else if (data_type == "uint8_t") {
    type_size = sizeof(uint8_t);
  } else if (data_type == "int32_t") {
    type_size = sizeof(int32_t);
  } else if (data_type == "uint32_t") {
    type_size = sizeof(uint32_t);
  } else {
    std::cerr << "Unsupported data type." << std::endl;
    exit(EXIT_FAILURE);
  }
  // allocate host and device buffers using the chosen type_size
  size_t total_elems = static_cast<size_t>(max_size) * static_cast<size_t>(max_size);
  size_t total_bytes = type_size * total_elems;

  A_raw = malloc(total_bytes);
  B_raw = malloc(total_bytes);
  C_raw = malloc(total_bytes);
  C_ref_raw = malloc(total_bytes);

  // get typed pointers from raw pointers based on the data type
  if (data_type == "float") {
    float *A = reinterpret_cast<float *>(A_raw);
    float *B = reinterpret_cast<float *>(B_raw);
    float *C = reinterpret_cast<float *>(C_raw);
    float *C_ref = reinterpret_cast<float *>(C_ref_raw);
    float *dA = reinterpret_cast<float *>(dA_raw);
    float *dB = reinterpret_cast<float *>(dB_raw);
    float *dC = reinterpret_cast<float *>(dC_raw);
    float *dC_ref = reinterpret_cast<float *>(dC_ref_raw);
  } else if (data_type == "double") {
    double *A = reinterpret_cast<double *>(A_raw);
    double *B = reinterpret_cast<double *>(B_raw);
    double *C = reinterpret_cast<double *>(C_raw);
    double *C_ref = reinterpret_cast<double *>(C_ref_raw);
    double *dA = reinterpret_cast<double *>(dA_raw);
    double *dB = reinterpret_cast<double *>(dB_raw);
    double *dC = reinterpret_cast<double *>(dC_raw); 
    double *dC_ref = reinterpret_cast<double *>(dC_ref_raw);
  } else if (data_type == "__half") {
    __half *A = reinterpret_cast<__half *>(A_raw);
    __half *B = reinterpret_cast<__half *>(B_raw);
    __half *C = reinterpret_cast<__half *>(C_raw);
    __half *C_ref = reinterpret_cast<__half *>(C_ref_raw);
    __half *dA = reinterpret_cast<__half *>(dA_raw);
    __half *dB = reinterpret_cast<__half *>(dB_raw);
    __half *dC = reinterpret_cast<__half *>(dC_raw);
    __half *dC_ref = reinterpret_cast<__half *>(dC_ref_raw);
  } else if (data_type == "__nv_bfloat16") {
    __nv_bfloat16 *A = reinterpret_cast<__nv_bfloat16 *>(A_raw);
    __nv_bfloat16 *B = reinterpret_cast<__nv_bfloat16 *>(B_raw);
    __nv_bfloat16 *C = reinterpret_cast<__nv_bfloat16 *>(C_raw);
    __nv_bfloat16 *C_ref = reinterpret_cast<__nv_bfloat16 *>(C_ref_raw);
    __nv_bfloat16 *dA = reinterpret_cast<__nv_bfloat16 *>(dA_raw);
    __nv_bfloat16 *dB = reinterpret_cast<__nv_bfloat16 *>(dB_raw);
    __nv_bfloat16 *dC = reinterpret_cast<__nv_bfloat16 *>(dC_raw);
    __nv_bfloat16 *dC_ref = reinterpret_cast<__nv_bfloat16 *>(dC_ref_raw);
  } else if (data_type == "int8_t") {
    int8_t *A = reinterpret_cast<int8_t *>(A_raw);
    int8_t *B = reinterpret_cast<int8_t *>(B_raw);
    int8_t *C = reinterpret_cast<int8_t *>(C_raw);
    int8_t *C_ref = reinterpret_cast<int8_t *>(C_ref_raw);
    int8_t *dA = reinterpret_cast<int8_t *>(dA_raw);
    int8_t *dB = reinterpret_cast<int8_t *>(dB_raw);
    int8_t *dC = reinterpret_cast<int8_t *>(dC_raw);
    int8_t *dC_ref = reinterpret_cast<int8_t *>(dC_ref_raw);
  } else if (data_type == "uint8_t") {
    uint8_t *A = reinterpret_cast<uint8_t *>(A_raw);
    uint8_t *B = reinterpret_cast<uint8_t *>(B_raw);
    uint8_t *C = reinterpret_cast<uint8_t *>(C_raw);
    uint8_t *C_ref = reinterpret_cast<uint8_t *>(C_ref_raw);
    uint8_t *dA = reinterpret_cast<uint8_t *>(dA_raw);
    uint8_t *dB = reinterpret_cast<uint8_t *>(dB_raw);
    uint8_t *dC = reinterpret_cast<uint8_t *>(dC_raw);
    uint8_t *dC_ref = reinterpret_cast<uint8_t *>(dC_ref_raw);
  } else if (data_type == "int32_t") {
    int32_t *A = reinterpret_cast<int32_t *>(A_raw);
    int32_t *B = reinterpret_cast<int32_t *>(B_raw);
    int32_t *C = reinterpret_cast<int32_t *>(C_raw);
    int32_t *C_ref = reinterpret_cast<int32_t *>(C_ref_raw);
    int32_t *dA = reinterpret_cast<int32_t *>(dA_raw);
    int32_t *dB = reinterpret_cast<int32_t *>(dB_raw);
    int32_t *dC = reinterpret_cast<int32_t *>(dC_raw);
    int32_t *dC_ref = reinterpret_cast<int32_t *>(dC_ref_raw);
  } else if (data_type == "uint32_t") {
    uint32_t *A = reinterpret_cast<uint32_t *>(A_raw);
    uint32_t *B = reinterpret_cast<uint32_t *>(B_raw);
    uint32_t *C = reinterpret_cast<uint32_t *>(C_raw);
    uint32_t *C_ref = reinterpret_cast<uint32_t *>(C_ref_raw);
    uint32_t *dA = reinterpret_cast<uint32_t *>(dA_raw);
    uint32_t *dB = reinterpret_cast<uint32_t *>(dB_raw);
    uint32_t *dC = reinterpret_cast<uint32_t *>(dC_raw);
    uint32_t *dC_ref = reinterpret_cast<uint32_t *>(dC_ref_raw);
  }
  
  // randomize host matrices
  randomize_matrix(A, total_elems);
  randomize_matrix(B, total_elems);
  randomize_matrix(C, total_elems);

  // device allocations
  cudaCheck(cudaMalloc(&dA, total_bytes));
  cudaCheck(cudaMalloc(&dB, total_bytes));
  cudaCheck(cudaMalloc(&dC, total_bytes));
  cudaCheck(cudaMalloc(&dC_ref, total_bytes));

  // copy host -> device
  cudaCheck(cudaMemcpy(dA, A, total_bytes, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dB, B, total_bytes, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC, C, total_bytes, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC_ref, C, total_bytes, cudaMemcpyHostToDevice));

  int repeat_times = 50;
  for (int size : SIZE) {
    m = n = k = size;

    std::cout << "dimensions(m=n=k) " << m << ", alpha: " << alpha
              << ", beta: " << beta << std::endl;
    // Verify the correctness of the calculation, and execute it once before the
    // kernel function timing to avoid cold start errors
    if (kernel_num != 0) {
      run_kernel(0, m, n, k, alpha, dA, dB, beta, dC_ref,
                 handle); // cuBLAS
      run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC,
                 handle); // Executes the kernel, modifies the result matrix
      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
      cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
      cudaMemcpy(C_ref, dC_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

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
        "Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
        "(%ld).\n",
        elapsed_time / repeat_times,
        (repeat_times * flops * 1e-9) / elapsed_time, m);
    fflush(stdout);
    // make dC and dC_ref equal again (we modified dC while calling our kernel
    // for benchmarking)
    cudaCheck(cudaMemcpy(dC, dC_ref, sizeof(float) * m * n,
                         cudaMemcpyDeviceToDevice));
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

  return 0;
};