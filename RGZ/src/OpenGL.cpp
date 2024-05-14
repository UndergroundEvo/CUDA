#include "RGZ.h"

GLchar *loadShader() {
    // Указываем файл и выбираем режим чтение
    std::ifstream file("Multiplication.comp.glsl", std::ifstream::ate);
    const int len = file.tellg();

    auto *computeSource = new GLchar[len + 1];

    file.seekg(0, file.beg);
    for (int i = 0; i < len; i++) file.get(computeSource[i]);

    computeSource[len] = '\0';
    file.close();
    return computeSource;
}