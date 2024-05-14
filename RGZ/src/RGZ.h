#ifndef RGZ_H
#define RGZ_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <random>
#include <fstream>
#include <chrono>

const int windowHeight = 800;
const int windowWidth = 600;
const int matrixSize = 1 << 13;

inline GLchar infoLog[512];

// Matrix.cpp
float* generateRandomMatrix(int width, int height);
void printMatrix(float* matrix, int width, int height);

//OpenGL.cpp
GLchar *loadShader();

#endif //RGZ_H
