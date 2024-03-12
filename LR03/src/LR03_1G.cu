#include <iostream>
#include <stdio.h>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

const int n = 1 << 20;

typedef std::chrono::milliseconds ms;
typedef std::chrono::nanoseconds ns;

__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
}

int main() {
    float elapsedTime;
    int blockSize = 1024;
    cudaEvent_t start, stop;
    chrono::time_point<chrono::system_clock> start_chrono, end_chrono;

    int numBlocks;
    cout << "Enter threads num: ";
    cin >> numBlocks;

    float *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, n * sizeof(float));
    cudaMalloc((void **) &d_b, n * sizeof(float));
    cudaMalloc((void **) &d_c, n * sizeof(float));

    float *h_a = new float[n],
            *h_b = new float[n];
    for (int i = 0; i < n; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    start_chrono = chrono::system_clock::now();
    vectorAdd<<<dim3(numBlocks), blockSize>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop, 0);
    end_chrono = chrono::system_clock::now();

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    float *h_c = new float[n];
    cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "CUDA Event time:\n\t"
         << elapsedTime
         << endl;

    cout << "Chrono time:\n\t"
         << chrono::duration_cast<ms>(end_chrono - start_chrono).count() << "ms\n\t"
         << chrono::duration_cast<ns>(end_chrono - start_chrono).count() << "ns"
         << endl;

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
