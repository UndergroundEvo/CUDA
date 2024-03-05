#include <iostream>
#include <cstdlib>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

#define CUDA_NUM 32

__global__ void gBase_Transposition(float *matrix, float *result, const int N, const int K) {
    unsigned int k = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int n = threadIdx.y + blockIdx.y * blockDim.y;
    result[n + k * N] = matrix[k + n * K];
}
__global__ void gShared_Transposition_Wrong(float *matrix, float *result, const int N, const int K) {
    __shared__ float shared[CUDA_NUM][CUDA_NUM];
    unsigned int k = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int n = threadIdx.y + blockIdx.y * blockDim.y;

    shared[threadIdx.y][threadIdx.x] = matrix[K + n * N];
    __syncthreads();

    k = threadIdx.x + blockIdx.y * blockDim.x;
    n = threadIdx.y + blockIdx.x * blockDim.y;
    result[k + n * N] = shared[threadIdx.x][threadIdx.y];
}
__global__ void gShared_Transposition(float *matrix, float *result, const int N, const int K) {
    __shared__ float shared[CUDA_NUM][CUDA_NUM + 1];
    unsigned int k = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int n = threadIdx.y + blockIdx.y * blockDim.y;

    shared[threadIdx.y][threadIdx.x] = matrix[K + n * N];
    __syncthreads();

    k = threadIdx.x + blockIdx.y * blockDim.x;
    n = threadIdx.y + blockIdx.x * blockDim.y;
    result[k + n * N] = shared[threadIdx.x][threadIdx.y];
}
void MatrixShow(const int N, const int K, const float *Matrix) {
    cout << endl;
    for (long long i = 0; i < K; ++i) {
        for (long long j = 0; j < N; ++j) {
            cout << Matrix[j + i * N] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

int main() {
    const int N = 8, K = 8, threadsPerBlock = 8;
    float *GPU_pre_matrix, *local_pre_matrix, *GPU_after_matrix, *local_after_matrix, elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* простое транспонирование */

    cudaMalloc((void **) &GPU_pre_matrix, N * K * sizeof(float));
    cudaMalloc((void **) &GPU_after_matrix, N * K * sizeof(float));

    local_pre_matrix = (float *) calloc(N * K, sizeof(float));
    local_after_matrix = (float *) calloc(N * K, sizeof(float));

    cout<<"Initial Matrix: "<<endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            local_pre_matrix[j + i * K] = j + i * K + 1;
            cout << local_pre_matrix[j + i * K] << " ";
        }
        cout<< endl;
    }
    cout<< endl;

    cudaMemcpy(GPU_pre_matrix, local_pre_matrix, K * N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start, nullptr);
    gBase_Transposition <<< dim3(K / threadsPerBlock,N / threadsPerBlock),
                            dim3(threadsPerBlock, threadsPerBlock) >>>
                            (GPU_pre_matrix, GPU_after_matrix, N, K);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);

    cudaMemcpy(local_after_matrix, GPU_after_matrix, K * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cout<<"1st method Matrix: "<<endl;
    MatrixShow(N, K, local_after_matrix);

    cout << "gBase_Transposition:\n\t"
         << elapsedTime
         << endl;

    cudaFree(GPU_after_matrix);
    free(local_after_matrix);

    /* траспонирование без решениея проблемы конфликта банков */

    cudaMalloc((void **) &GPU_after_matrix, N * K * sizeof(float));
    local_after_matrix = (float *) calloc(N * K, sizeof(float));

    cudaEventRecord(start, nullptr);
    gShared_Transposition_Wrong <<< dim3(K / threadsPerBlock, N / threadsPerBlock),
                                    dim3(threadsPerBlock,threadsPerBlock) >>>
                                    (GPU_pre_matrix, GPU_after_matrix, N, K);

    cudaDeviceSynchronize();
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);

    cudaMemcpy(local_after_matrix, GPU_after_matrix, K * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cout<<"2st method Matrix: "<<endl;
    MatrixShow(N, K, local_after_matrix);

    cout << "gShared_Transposition_Wrong:\n\t"
         << elapsedTime
         << endl;

    cudaFree(GPU_after_matrix);
    free(local_after_matrix);

    /* траспонирование с решением проблемы конфликта банков */

    cudaMalloc((void **) &GPU_after_matrix, N * K * sizeof(float));
    local_after_matrix = (float *) calloc(N * K, sizeof(float));

    cudaEventRecord(start, nullptr);
    gShared_Transposition <<< dim3(K / threadsPerBlock, N / threadsPerBlock),
                              dim3(threadsPerBlock,threadsPerBlock) >>>
                              (GPU_pre_matrix, GPU_after_matrix, N, K);

    cudaDeviceSynchronize();
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);

    cudaMemcpy(local_after_matrix, GPU_after_matrix, K * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cout<<"3st method Matrix: "<<endl;
    MatrixShow(N, K, local_after_matrix);

    cout << "gShared_Transposition:\n\t"
         << elapsedTime
         << endl;

    cudaFree(GPU_pre_matrix);
    cudaFree(GPU_after_matrix);
    free(local_pre_matrix);
    free(local_after_matrix);

    return 0;
}