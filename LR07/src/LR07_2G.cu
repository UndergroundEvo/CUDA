#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <iomanip>

#define CUDA_NUM 32

using namespace std;

__global__ void gFunc(int *A, int N, int K) {
    __shared__ int B[CUDA_NUM][CUDA_NUM + 1];
    unsigned int k = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int n = threadIdx.y + blockIdx.y * blockDim.y;
    if (k >= N || n >= K) return;

    B[threadIdx.y][threadIdx.x] = A[n + k * K];
    __syncthreads();
    A[k + n * N] = B[threadIdx.y][threadIdx.x];
}


int main() {
    const int num = 1 << 12;
    int N = 4 * num, K = 8 * num, threads_per_block = 128;
    float elapsedTime = 0;
    cudaEvent_t start, stop;

    int *map = (int *) calloc(N * K, sizeof(int));
    int *A = (int *) calloc(N * K, sizeof(int));

    int *hA;
    cudaMalloc((void **) &hA, K * N * sizeof(int));

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            A[j + i * K] = j + i * K;
        }
    }

    cudaMemcpy(hA, A, N * K * sizeof(int), cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    gFunc<<<dim3(threads_per_block, threads_per_block),
            dim3((N + threads_per_block - 1) / threads_per_block,(K + threads_per_block - 1) / threads_per_block)
            >>>(hA, N,K);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "Time using raw CUDA code: " << setprecision(15) << elapsedTime << endl;
    elapsedTime = 0;

    cudaMemcpy(A, hA, N * K * sizeof(float), cudaMemcpyDeviceToHost);

    free(A);
    cudaFree(hA);

    thrust::host_vector<int> vA(N * K);
    for (int i = 0; i < N * K; ++i) {
        vA[i] = i;
    }
    thrust::device_vector<int> dA = vA;
    thrust::device_vector<int> dA_T(N * K);


    for (int i = 0; i < K * N; ++i) {
        map[i] = (i % N) * K + (i / N);
    }

    thrust::device_vector<int> d_map(map, map + K * N);


    cudaEventRecord(start, 0);
    thrust::gather(d_map.begin(), d_map.end(), dA.begin(), dA_T.begin());
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "Time using Thrust lib: " << setprecision(15) << elapsedTime << endl;
}
