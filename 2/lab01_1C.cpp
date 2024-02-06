//
// Created by miron on 06.02.24.
//

#include <iostream>
#include <stdlib.h>

using namespace std;
const long long N = 99999999;

void vectorAdd(const float *a, const float *b, float *c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    float *a, *b, *c;
    a = new float[N];
    b = new float[N];
    c = new float[N];

    for (int i = 0; i < N; ++i) {
        a[i] = rand()%9999999 +1;
        b[i] = rand()%9999999 +1;
    }
    cout<<"числа с генерированы"<<endl;

    vectorAdd(a, b, c, N);

    for (int i = 0; i < 10; ++i) {
        std::cout << "c[" << i << "] = " << c[i] << std::endl;
    }

    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}




