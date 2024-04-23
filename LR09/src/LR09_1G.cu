#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;

int main() {
    int num = 1 << 12;
    int size = 32 * num;
    float *device, *hostPinned, *host, time = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    host = (float *) malloc(size * sizeof(float));
    cudaMallocHost((void **) &hostPinned, size * sizeof(float));
    cudaMalloc((void **) &device, size * sizeof(float));
    cudaMemset(device, 1024, size * sizeof(float));
    cudaMemcpy(device, host, size * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start, nullptr);
    cudaMemcpy(host, device, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cout << "Стандартное копирование с device на host:             "<< time << endl;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEventRecord(start, nullptr);
    cudaMemcpyAsync(hostPinned, device, size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);
    time = 0;
    cudaEventElapsedTime(&time, start, stop);
    cout << "Закрепленная память (pinned memory) с device на хост: "<< time  << endl;

    cudaEventRecord(start, nullptr);
    cudaMemcpy(device, host, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);
    time = 0;
    cudaEventElapsedTime(&time, start, stop);
    cout << "Стандартное копирование с host на device:             " << time << endl;

    cudaEventRecord(start, nullptr);
    cudaMemcpyAsync(device, hostPinned, size * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);
    time = 0;
    cudaEventElapsedTime(&time, start, stop);
    cout << "Закрепленная память (pinned memory) с хост на device: " << time << endl;

    cudaFree(device);
    cudaFreeHost(host);
    free(host);
    return 0;
}

