#include <cuda.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;
// https://habr.com/ru/articles/54707/

__global__ void gTransposition(int *a, int *b, int N, int K) {
    //extern __shared__ int B[];
    unsigned int k = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int n = threadIdx.y + blockIdx.y * blockDim.y;
    b[n + k * N] = a[k + n * K];
    //B[n + k * N] = a[k + n * K];
    //__syncthreads();
    //b[k + n * N] = B[k + n * N];
}

int main() {
    const int N = 4, K = 8, threads_per_block = 4;
    float elapsedTime;
    int *GPU_pre_matrix, *local_pre_matrix, *GPU_after_matrix, *local_after_matrix;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc((void **) &GPU_pre_matrix, N * K * sizeof(int));
    cudaMalloc((void **) &GPU_after_matrix, N * K * sizeof(int));
    local_pre_matrix = (int *) calloc(N * K, sizeof(int));
    local_after_matrix = (int *) calloc(N * K, sizeof(int));

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            local_pre_matrix[j + i * K] = j + i * K + 1;
            cout << local_pre_matrix[j + i * K] << " ";
        }
        cout<< endl;
    }

    cout << "\ndim3((K + threads_per_block - 1) / threads_per_block = "
         << ((K + threads_per_block - 1) / threads_per_block,(N + threads_per_block - 1) / threads_per_block) << "\n"
         <<"x = " << dim3((K + threads_per_block - 1) / threads_per_block,(N + threads_per_block - 1) / threads_per_block).x << "\t"
         <<"y = " << dim3((K + threads_per_block - 1) / threads_per_block,(N + threads_per_block - 1) / threads_per_block).y << "\t"
         <<"z = " << dim3((K + threads_per_block - 1) / threads_per_block,(N + threads_per_block - 1) / threads_per_block).z << "\t\n"
         << "dim3(threads_per_block, threads_per_block) = "
         << threads_per_block << "\n"
         <<"x = " << dim3(threads_per_block, threads_per_block).x << "\t"
         <<"y = " << dim3(threads_per_block, threads_per_block).y << "\t"
         <<"z = " << dim3(threads_per_block, threads_per_block).z << endl;

    cudaMemcpy(GPU_pre_matrix, local_pre_matrix, K * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(start, nullptr);

    gTransposition <<< dim3((K + threads_per_block - 1) / threads_per_block,(N + threads_per_block - 1) / threads_per_block),
                       dim3(threads_per_block, threads_per_block)>>>(GPU_pre_matrix, GPU_after_matrix, N, K);

    cudaDeviceSynchronize();

    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);
    cudaMemcpy(local_after_matrix, GPU_after_matrix, K * N * sizeof(float), cudaMemcpyDeviceToHost);

    cout << endl;
    for (long long i = 0; i < K; ++i) {
        for (long long j = 0; j < N; ++j) {
            cout << local_after_matrix[j + i * N] << " ";
        }
        cout << endl;
    }
    cout << endl;

    cudaEventElapsedTime(&elapsedTime, start, stop);

    cout << "CUDA Event time:\n\t"
         << elapsedTime
         << endl;

    cudaFree(GPU_pre_matrix);
    cudaFree(GPU_after_matrix);
    free(local_pre_matrix);
    free(local_after_matrix);

    return 0;
}