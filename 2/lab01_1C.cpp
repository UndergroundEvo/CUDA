#include <iostream>
#include <vector>
#include <thread>
#include <time.h>

void vectorAdd(const std::vector<float> &a, const std::vector<float> &b, std::vector<float> &c, int start, int end) {
    for (int i = start; i < end; ++i) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 10000;
    std::vector<float> a(n), b(n), c(n);
    for (int i = 0; i < n; ++i) {
        a[i] = i;
        b[i] = i * 2;
    }
    //количество поток создаеться изходя из потоков пороцессора
    int numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; ++i) {
        int start = i * (n / numThreads);
        int end = (i == numThreads - 1) ? n : (i + 1) * (n / numThreads);
        threads.emplace_back(vectorAdd, std::ref(a), std::ref(b), std::ref(c), start, end);
    }

    for (auto &thread : threads) {
        thread.join();
    }

    // Выводим результат
//    std::cout << "Result: ";
//    for (int i = 0; i < n; ++i) {
//        std::cout << c[i] << " ";
//    }
//    std::cout << std::endl;

    return 0;
}