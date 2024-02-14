#include <iostream>

// CUDA ядро для сложения векторов на GPU
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    // Размер векторов
    int n = 10000;

    // Выделяем память под векторы на устройстве (GPU)
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, n * sizeof(float));
    cudaMalloc((void **)&d_b, n * sizeof(float));
    cudaMalloc((void **)&d_c, n * sizeof(float));

    // Заполняем векторы на хосте (CPU)
    float *h_a = new float[n];
    float *h_b = new float[n];
    for (int i = 0; i < n; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Копируем данные из памяти на хосте в память на устройстве
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

    // Вычисляем количество блоков и потоков на блок
    int blockSize = 4096;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Вызываем CUDA ядро для сложения векторов на GPU
    vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);

    // Копируем результат обратно на хост
    float *h_c = new float[n];
    cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Выводим результат
//    std::cout << "Result: ";
//    for (int i = 0; i < n; ++i) {
//        std::cout << h_c[i] << " ";
//    }
//    std::cout << std::endl;

    // Освобождаем память
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
