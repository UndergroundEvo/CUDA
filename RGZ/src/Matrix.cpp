#include "RGZ.h"

float* generateRandomMatrix(const int width,const int height) {
    std::random_device rd;
    // Выбор генератора для рандома, в данном случае Вихрь Мерсенна
    std::mt19937 gen(rd());
    // Генерация чисел в диапазоне 0.0, 1.0
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    auto* matrix = new float[width * height];
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            matrix[i * width + j] = dis(gen);
        }
    }
    return matrix;
}
void printMatrix(float* matrix,const int width,const int height) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << matrix[i * width + j] << " ";
        }
        std::cout << std::endl;
    }
}