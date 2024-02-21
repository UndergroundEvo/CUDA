#include <iostream>
#include <chrono>
#include <vector>
#include <thread>

using namespace std;

const int n = 100000000;

typedef std::chrono::milliseconds ms;
typedef std::chrono::nanoseconds ns;

void vectorAdd(const vector<float> &a, const vector<float> &b, vector<float> &c, int start, int end)
{
    for (int i = start; i < end; ++i)
        c[i] = a[i] + b[i];
}

int main()
{
    vector<float> a(n), b(n), c(n);
    int numThreads = thread::hardware_concurrency();

    chrono::time_point<chrono::system_clock> start, end;

    for (int i = 0; i < n; ++i)
    {
        a[i] = i;
        b[i] = i * 2;
    }

    vector<thread> threads;
    for (int i = 0; i < numThreads; ++i)
    {
        int start = i * (n / numThreads),
            end = (i == numThreads - 1) ? n : (i + 1) * (n / numThreads);
        threads.emplace_back(vectorAdd, ref(a), ref(b), ref(c), start, end);
    }

    start = chrono::system_clock::now();
    for (auto &thread : threads)
        thread.join();
    end = chrono::system_clock::now();

    cout << "Wasted time:\n\t"
         << chrono::duration_cast<ms>(end - start).count() << "ms\n\t"
         << chrono::duration_cast<ns>(end - start).count() << "ns"
         << endl;

    return 0;
}
