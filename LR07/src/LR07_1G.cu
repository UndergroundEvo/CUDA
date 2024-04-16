#include <thrust/generate.h>
#include <thrust/gather.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cstdlib>
#include <iostream>
#include <iomanip>

using namespace std;

__global__ void gFunc(int *A, int *B, int *C, int N) {
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N) return;
    C[i] = A[i] + B[i];
}

int main() {
    const int num = 1 << 13;
    int N = 4 * num, K = 8 * num, threads_per_block = 128;
    float elapsedTime = 0;
    cudaEvent_t start, stop;

    int *hA, *hB, *hC;
    int *A = (int *) calloc(N * K, sizeof(int));
    int *B = (int *) calloc(N * K, sizeof(int));
    int *C = (int *) calloc(N * K, sizeof(int));

    cudaMalloc((void **) &hA, N * K * sizeof(int));
    cudaMalloc((void **) &hB, N * K * sizeof(int));
    cudaMalloc((void **) &hC, N * K * sizeof(int));

    for (int i = 0; i < N; ++i) {
        A[i] = i;
        B[i] = i + 1;
    }

    cudaMemcpy(hA, A, N * K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(hB, B, N * K * sizeof(int), cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    gFunc<<<dim3(threads_per_block),
    dim3((N * K + threads_per_block - 1) / threads_per_block)
    >>>(hA, hB, hC, N * K);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "Time using raw CUDA code: " << setprecision(15) << elapsedTime << endl;
    elapsedTime = 0;

    cudaMemcpy(C, hC, N * K * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(hA);
    cudaFree(hB);
    cudaFree(hC);

    thrust::host_vector<int> vA(A, A + N * K);
    thrust::host_vector<int> vB(B, B + N * K);
    thrust::host_vector<int> vC(N * K);

    thrust::device_vector<int> dA = vA;
    thrust::device_vector<int> dB = vB;
    thrust::device_vector<int> dC(N * K);

    cudaEventRecord(start, 0);
    thrust::transform(dA.begin(), dA.end(), dB.begin(), dC.begin(), thrust::multiplies<int>());
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "Time using Thrust lib: " << setprecision(15) << elapsedTime << endl;
}