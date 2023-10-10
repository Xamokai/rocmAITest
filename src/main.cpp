#include <hip/hip_runtime.h>
#include "utils.h"
#include <array>

__global__ void train(int i, int nets, arr* d_weights, arr* d_biases){
  unsigned long a = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned long b = a + i;

}

int main(){
  unsigned int num_nets = 256;

  unsigned int layout[] = {9, 5, 5, 9};
  unsigned int num_biases = 0;
  unsigned int num_weights = 0;

  for(int i = 0; i < ((sizeof(layout) / sizeof(int)) - 1); i++){
    num_biases += layout[i];
    num_weights += layout[i] * layout[i+1];
  }
  
  array<float> d_weights;
  float* d_biases;

  HIP_CHECK(hipMalloc(&d_weights, sizeof(float) * num_weights * num_nets));
  HIP_CHECK(hipMalloc(&d_biases, sizeof(float) * num_biases * num_nets));

  int iterations = 200;
  for(int i = 0; i < iterations; i++){
    train<<<dim3(2), dim3(64)>>>(i, num_nets, d_weights, d_biases);
    hipDeviceSynchronize();
  }
}
