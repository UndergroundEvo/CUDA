//
// Created by miron on 06.02.24.
//

#include <iostream>
#include <vector>
#include <thread>
using namespace std;

const int n = 10000;

void vectorAdd(const vector<float> &a, const vector<float> &b, vector<float> &c, int start, int end) {
    for (int i = start; i < end; ++i) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    vector<float> a(n), b(n), c(n);
    int numThreads = thread::hardware_concurrency();

    for (int i = 0; i < n; ++i) {
        a[i] = rand()%9999999 +1;;
        b[i] = rand()%9999999 +1;;
    }

    vector<thread> threads;
    for (int i = 0; i < numThreads; ++i) {
        int start = i * (n / numThreads);
        int end = (i == numThreads - 1) ? n : (i + 1) * (n / numThreads);
        threads.emplace_back(vectorAdd, ref(a), ref(b), ref(c), start, end);
    }

    for (auto &thread : threads) {
        thread.join();
    }

    cout << "Result: ";
    for (int i = 0; i < n; ++i) {
        cout << c[i] << " \n";
    }
    cout << endl;
    return 0;
}
