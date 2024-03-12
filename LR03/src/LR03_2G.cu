#include <iostream>
#include <cstdlib>
#include <cuda.h>

#include "device_launch_parameters.h"

using namespace std;

const int n = 1 << 20;

__global__ void vectorAdd(int arr1[], int arr2[]) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    arr1[i] += arr2[i];
}

int main() {
    int *local_arr1, *local_arr2, *GPU_arr1, *GPU_arr2;
    int threads_per_block, num_of_blocks;
    float time;

    cout << "Enter threads num: ";
    cin >> threads_per_block;
    //threads_per_block = 2048;
    num_of_blocks = n / threads_per_block;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc((void **) &GPU_arr1, n * sizeof(int));
    cudaMalloc((void **) &GPU_arr2, n * sizeof(int));
    local_arr1 = (int *) calloc(n, sizeof(int));
    local_arr2 = (int *) calloc(n, sizeof(int));


    for (int i = 0; i < n; i++) {
        local_arr1[i] = i;
        local_arr2[i] = i + 1;
    }

    cudaMemcpy(GPU_arr1, local_arr1, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(GPU_arr2, local_arr2, n * sizeof(int), cudaMemcpyHostToDevice);

    cudaEventRecord(start, nullptr);
    vectorAdd<<<dim3(num_of_blocks),dim3(threads_per_block)>>>(GPU_arr1, GPU_arr2);
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);

    cout << "CUDA Event time:\n\t"
         << time <<"ns"
         << endl;

    cudaFree(GPU_arr1);
    cudaFree(GPU_arr2);
    delete[] local_arr1;
    delete[] local_arr2;

    return 0;
}