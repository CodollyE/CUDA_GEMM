#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#define cudaCheck(call) cuda_check((call), __FILE__, __LINE__)

inline void cuda_check(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("CUDA error at %s:%d: %s\n", file, line,
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

template<typename T>
void runner(int kernel_num); // The main function to run the kernel and test
void CudaDeviceInfo();    // print CUDA information

void range_init_matrix(float *mat, int N);
void randomize_matrix(float *mat, int N);
void zero_init_matrix(float *mat, int N);
void copy_matrix(const float *src, float *dest, int N);
void print_matrix(const float *A, int M, int N, std::ofstream &fs);
bool verify_matrix(float *mat1, float *mat2, int N);

float get_current_sec();                        // Get the current moment
float cpu_elapsed_time(float &beg, float &end); // Calculate time difference

void run_kernel(int kernel_num, int m, int n, int k, float alpha, float *A,
                float *B, float beta, float *C, cublasHandle_t handle);