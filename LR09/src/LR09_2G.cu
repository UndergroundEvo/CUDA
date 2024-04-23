#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

void show_mass(float *a, int num){
    for (int i = 0; i < num; i++) {
        printf("%f ",a[i]);
        if (i%10 == 0) printf("\n");
    }
    printf("\n");
}
__global__ void addVectors(float *a, float *b, float *c, int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        c[i] = a[i] + b[i];
    }
}

int main() {
    int stream_num = 1;
    int num = 1 << 12;
    int size = 32 * num;
    int portion_size = size / stream_num;

    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    float time = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto *streams = (cudaStream_t*)calloc(stream_num, sizeof(cudaStream_t));
    for (int i = 0; i < stream_num; i++) cudaStreamCreate(&streams[i]);

    cudaMallocHost((void **) &h_a, size * sizeof(float));
    cudaMallocHost((void **) &h_b, size * sizeof(float));
    cudaMallocHost((void **) &h_c, size * sizeof(float));
    cudaMalloc((void **)&d_a, size * sizeof(float));
    cudaMalloc((void **)&d_b, size * sizeof(float));
    cudaMalloc((void **)&d_c, size * sizeof(float));

    for (int i = 0; i < size; i++) {
        h_a[i] = i;
        h_b[i] = i + 1;
    }

    for (int i = 0; i < stream_num; i++) {
        cudaMemcpyAsync(d_a + i * portion_size, h_a + i * portion_size, portion_size * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_b + i * portion_size, h_b + i * portion_size, portion_size * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
    }
    for (int i = 0; i < stream_num; i++) cudaStreamSynchronize(streams[i]);

    dim3 threadsPerBlock(256);
    dim3 numBlocks((portion_size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    cudaEventRecord(start, nullptr);
    for (int i = 0; i < stream_num; i++)
        addVectors<<<numBlocks, threadsPerBlock, 0, streams[i]>>>(d_a + i * portion_size, d_b + i * portion_size, d_c + i * portion_size, portion_size);

    for (int i = 0; i < stream_num; i++) {
        cudaMemcpyAsync(h_c + i * portion_size, d_c + i * portion_size, portion_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }
    for (int i = 0; i < stream_num; i++) cudaStreamSynchronize(streams[i]);

    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cout << "time = "<< time << endl;

    //show_mass(h_c, 100);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(h_a);
    cudaFree(h_b);
    cudaFree(h_c);
    for (int i = 0; i < stream_num; i++) cudaStreamDestroy(streams[i]);

    return 0;
}
