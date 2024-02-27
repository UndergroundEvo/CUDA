#include <iostream>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

const unsigned int N = 10;
const unsigned int K = 10;
const unsigned int BANK_SIZE = 32;
const unsigned int blockSize = 1024;
const unsigned int numBlocks = 2048;

typedef std::chrono::milliseconds ms;
typedef std::chrono::nanoseconds ns;

__global__ void MatrixFill(int *a1) {
    unsigned int k = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int n = blockDim.y * blockIdx.y + threadIdx.y;
//  if (k >= N || n >= K) return;
//  a1[k + N * n] = k + N * n;
//  for (int i=0;i<N;i++) a1[k + N * n] = i;
    a1[k + N * n] = k + N * n;
}
__global__ void TransMatrix(int *a1, int *a2) {
    unsigned int k = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int n = blockDim.y * blockIdx.y + threadIdx.y;
    if (k >= N || n >= K) return;
    a2[n + K * k] = a1[k + N * n];
}

int main() {
    float elapsedTime;
    cudaEvent_t start, stop;
    chrono::time_point<chrono::system_clock> start_chrono, end_chrono;

    int *a1, *a2, final[K*N];
    cudaMalloc((void **) &a1, N*K * sizeof(int));
    cudaMalloc((void **) &a2, N*K * sizeof(int));

    dim3 dim_block(BANK_SIZE, BANK_SIZE);
    dim3 dim_grid((K - 1) / BANK_SIZE + 1, (N - 1) / BANK_SIZE + 1);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    start_chrono = chrono::system_clock::now();

//    gInit<<<numBlocks, blockSize>>>(a1);
//    gCopy<<<numBlocks, blockSize>>>(a1, a2);
    MatrixFill<<<dim_grid, dim_block>>>(a1);
    TransMatrix<<<dim_grid, dim_block>>>(a1, a2);
//    gInit<<<N,N>>>(a1);
//    gCopy<<<N,N>>>(a1, a2);

    cudaEventRecord(stop, 0);
    end_chrono = chrono::system_clock::now();
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaMemcpy(final, a1, N*K * sizeof(int), cudaMemcpyDeviceToHost);

    cout << "CUDA Event time:\n\t"
         << elapsedTime
         << endl;

    cout << "Chrono time:\n\t"
         << chrono::duration_cast<ms>(end_chrono - start_chrono).count() << "ms\n\t"
         << chrono::duration_cast<ns>(end_chrono - start_chrono).count() << "ns"
         << endl;

    cudaMemcpy(final, a1, N*K * sizeof(int), cudaMemcpyDeviceToHost);
    cout << "Non transpose:  \n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < 10; j++) cout << " " << final[j + i * N];
        cout << "\n";
    }
    cout << "\n";

    cudaMemcpy(final, a2, N * sizeof(int), cudaMemcpyDeviceToHost);
    cout << "Sorted: \n";
    for(int i = 0 ; i < N ; i++){
        for (int j = 0 ; j < 10 ; j ++) cout << " " << final[j + i * N];
        cout << "\n";
    }
    cout << "\n";

    cudaFree(a2);
    cudaFree(a1);

    return 0;
}
