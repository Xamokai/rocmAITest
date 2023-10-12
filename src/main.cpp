#include <hip/hip_runtime.h>
#include "utils.h"
#include <array>

struct arr{
  float* ptr;
  unsigned long size;

  __device__ __host__ float operator[](unsigned long i){
    return this->ptr[i];
  }

  __device__ __host__ arr slice(unsigned long i, unsigned long j){
    if(i < j) {
      return arr{nullptr, 0};
    }

    return arr{(this->ptr)+i, i - j};
  }
};

__device__ int checkBoard(int board[9]){
  int ret = 0;
  for(int i = 0; i < 9; i+=3){
    ret = ((board[0 + i] == board[1 + i]) && (board[1 + i] == board[2 + i])) * board[0 + i];
  }
  for(int i = 0; i < 3; i++){
    ret = ((board[0 + i] == board[3 + i]) && (board[3 + i] == board[6 + i])) * board[0 + i];
  }
  ret = ((board[0] == board[4]) && (board[4] == board[8])) * board[0];
  ret = ((board[2] == board[4]) && (board[4] == board[6])) * board[6];
  return ret;
}


__global__ void train(int i, int nets, arr d_weights, arr d_biases){
  unsigned long a = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned long b = (a + i) & (nets-1);
  if(a==b)
    b+=1;

  arr slice_a_w = d_weights.slice(a * d_weights.size, d_weights.size);
  arr slice_b_w = d_weights.slice(b * d_weights.size, d_weights.size);

  arr slice_a_b = d_biases.slice(a * d_biases.size, d_biases.size);
  arr slice_b_b = d_biases.slice(b * d_biases.size, d_biases.size);

  int board[9] = {0};
  
  int check = 0;
  for(int i = 0; i < 9 && checkBoard(board) == 0; i++){

    check = checkBoard(board);
  }

  arr from_w;
  arr to_w;
  if(check == 1){
    from_w = slice_a_w;
    to_w = slice_b_w;
  }else {
    from_w = slice_b_w;
    to_w = slice_a_w;
  }


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
  
  arr d_weights;
  arr d_biases;

  HIP_CHECK(hipMalloc(&d_weights.ptr, sizeof(float) * num_weights * num_nets));
  HIP_CHECK(hipMalloc(&d_biases.ptr, sizeof(float) * num_biases * num_nets));

  d_weights.size = num_weights;
  d_biases.size = num_biases;

  int iterations = 200;
  for(int i = 0; i < iterations; i++){
    train<<<dim3(2), dim3(64)>>>(i, num_nets, d_weights, d_biases);
    hipDeviceSynchronize();
  }
}
