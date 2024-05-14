#version 430 core
layout(local_size_x = 1, local_size_y = 1) in; // Указываем размер локальной рабочей группы

layout(std430, binding = 0) buffer matrixA_buffer{
    float matrixA[];
}; // Буфер для матрицы A

layout(std430, binding = 1) buffer matrixB_buffer{
    float matrixB[];
}; // Буфер для матрицы B

layout(std430, binding = 2) buffer matrixResult_buffer{
    float matrixResult[];
}; // Буфер для результирующей матрицы

uniform int matrixSize; // Размер матриц

void main(){
    // Получаем индексы элемента матрицы
    uint i = gl_GlobalInvocationID.x;
    uint j = gl_GlobalInvocationID.y;

    // Вычисляем произведение матриц
    float sum = 0.0;
    for (int k = 0; k < matrixSize; ++k) {
        sum += matrixA[i * matrixSize + k] * matrixB[k * matrixSize + j];
    }
    matrixResult[i * matrixSize + j] = sum;
}