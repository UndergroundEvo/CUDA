//
// Created by miron on 06.02.24.
//


#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
using namespace std;
const long long N = 99999999;

__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;

    cudaMalloc((void **)&d_a, N * sizeof(float));
    cudaMalloc((void **)&d_b, N * sizeof(float));
    cudaMalloc((void **)&d_c, N * sizeof(float));

    a = (float *)malloc(N * sizeof(float));
    b = (float *)malloc(N * sizeof(float));
    c = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        a[i] = rand()%9999999 +1;
        b[i] = rand()%9999999 +1;
    }
    cout<<"числа с генерированы"<<endl;

    // Копирование данных с хоста на устройство
    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

    // Выполнение ядра CUDA для сложения векторов
    vectorAdd<<<(N + 255) / 256, 256>>>(d_a, d_b, d_c, N);

    // Копирование результата с устройства на хост
    cudaMemcpy(c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; ++i) {
        std::cout << "c[" << i << "] = " << c[i] << std::endl;
    }

    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}




