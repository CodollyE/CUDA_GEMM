#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include "runner.cuh"
#include <vector>

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

  // get environment variable for device
  int deviceIdx = 0; // gpu id, device指gpu
  if (getenv("DEVICE") != NULL) {
    deviceIdx = atoi(getenv("DEVICE"));
  }
  cudaCheck(cudaSetDevice(deviceIdx));

  printf("Running kernel %d on device %d, dataType %s\n", kernel_num, deviceIdx, data_type.c_str());

  // print some device info
  // CudaDeviceInfo();

  // run the kernel and test
  if (data_type == "float") {
    runner<float>(kernel_num);
  } else if (data_type == "double") {
    runner<double>(kernel_num);
  } else if (data_type == "__half") {
    runner<__half>(kernel_num);
  } else if (data_type == "__nv_bfloat16") {
    runner<__nv_bfloat16>(kernel_num);
  } else if (data_type == "int8_t") {
    runner<int8_t>(kernel_num);
  } else if (data_type == "uint8_t") {
    runner<uint8_t>(kernel_num);
  } else if (data_type == "int32_t") {
    runner<int32_t>(kernel_num);
  } else if (data_type == "uint32_t") {
    runner<uint32_t>(kernel_num);
  } else {
    std::cerr << "Unsupported data type." << std::endl;
    exit(EXIT_FAILURE);
  }
  return 0;
};