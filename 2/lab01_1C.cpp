//
// Created by miron on 06.02.24.
//

#include <iostream>
const long long N = 999989; // Размер векторов

void vectorAdd(const float *a, const float *b, float *c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    float *a, *b, *c;   // Векторы на хосте

    // Выделение памяти на хосте
    a = new float[N];
    b = new float[N];
    c = new float[N];

    // Заполнение векторов на хосте
    for (int i = 0; i < N; ++i) {
        a[i] = i;
        b[i] = i * 2;
    }

    // Выполнение сложения векторов на CPU
    vectorAdd(a, b, c, N);

    // Вывод результата
    for (int i = 0; i < 10; ++i) {
        std::cout << "c[" << i << "] = " << c[i] << std::endl;
    }

    // Освобождение памяти
    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}




