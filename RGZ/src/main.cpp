#include "RGZ.h"

typedef std::chrono::minutes min;
typedef std::chrono::seconds sec;


int main() {
    std::chrono::time_point<std::chrono::system_clock> start, end;

    // Проверка на инициализацию GLFW (графическая часть OpenGL)
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Устанавливаем параметры окна GLFW
    //glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Создаем GLFW окно
    GLFWwindow* window = glfwCreateWindow(windowWidth, windowHeight, "Matrix muliplication", nullptr, nullptr);
    if (window == nullptr) {
        std::cerr << "Failed to create GLFW window" << glfwGetError(NULL) << std::endl;
        //glfwTerminate();
        //return -1;
    }

    // Загружаем вычислительный шейдер из файла "Multiplication.comp.glsl"
    char* computeShaderSource = loadShader();

    // Выбираем наше созданное окно
    glfwMakeContextCurrent(window);

    // Проверка на инициализацию GLEW (расширение для OpenGL)
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW: " << glewGetErrorString(glewInit()) << std::endl;
        glfwTerminate();
        return -1;
    }

    // Создаем и загружаем вычислительный шейдер
    GLuint computeShader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(computeShader, 1, &computeShaderSource, nullptr);
    glCompileShader(computeShader);

    // Компилируем шейдер и проверяем его
    GLint success;
    glGetShaderiv(computeShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(computeShader, 512, nullptr, infoLog);
        std::cerr << "Compute shader compilation failed\n" << infoLog << std::endl;
        glfwTerminate();
        return -1;
    }

    // Создаем программу и приклепляем к ней шейдер
    GLuint computeProgram = glCreateProgram();
    glAttachShader(computeProgram, computeShader);
    glLinkProgram(computeProgram);

    glGetProgramiv(computeProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(computeProgram, 512, nullptr, infoLog);
        std::cerr << "Compute program linking failed\n" << infoLog << std::endl;
        glfwTerminate();
        return -1;
    }

    // Создаем и генерируем матрицы
    float* matrixA = generateRandomMatrix(matrixSize, matrixSize);
    float* matrixB = generateRandomMatrix(matrixSize, matrixSize);

    // Инициализируем под каждую матрицу буфер
    GLuint matrixBufferA, matrixBufferB, matrixBufferResult;
    glGenBuffers(1, &matrixBufferA); // генерация буфера
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, matrixBufferA); // привязка буфера, указываем что это шейдерный буфер
    // Указываем размеры буфера, указываем что он статический
    glBufferData(GL_SHADER_STORAGE_BUFFER, matrixSize * matrixSize * sizeof(float), nullptr, GL_STATIC_DRAW);

    glGenBuffers(1, &matrixBufferB);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, matrixBufferB);
    glBufferData(GL_SHADER_STORAGE_BUFFER, matrixSize * matrixSize * sizeof(float), nullptr, GL_STATIC_DRAW);

    glGenBuffers(1, &matrixBufferResult);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, matrixBufferResult);
    glBufferData(GL_SHADER_STORAGE_BUFFER, matrixSize * matrixSize * sizeof(float), nullptr, GL_STATIC_DRAW);

    // Копируем матрицы с хоста на девайс
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, matrixBufferA);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, matrixSize * matrixSize * sizeof(float), matrixA);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, matrixBufferB);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, matrixSize * matrixSize * sizeof(float), matrixB);


    start = std::chrono::system_clock::now();
    // Выбираем программу
    glUseProgram(computeProgram);
    // Привязываем буферный шейдер
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, matrixBufferA);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, matrixBufferB);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, matrixBufferResult);

    //загружаем программу и подаем в неё обьект программы и указываем входной параметр
    glUniform1i(glGetUniformLocation(computeProgram, "matrixSize"), matrixSize);
    glDispatchCompute(matrixSize, matrixSize, 1);

    // Ждем завершение работы программы
    // для этого ставим барьер типо cudaDeviceSynchronize();
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // Копируем с девайса на хост результат
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, matrixBufferResult);
    auto* matrixResult = (float*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
    end = std::chrono::system_clock::now();

    if (matrixResult) {
        std::cout << "Result Matrix:" << std::endl;
        //printMatrix(matrixResult, matrixSize, matrixSize);
        printMatrix(matrixResult, 32, 32);
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    }

    std::cout << "Wasted time:\n\t"
         << std::chrono::duration_cast<min>(end - start).count() << "min\n\t"
         << std::chrono::duration_cast<sec>(end - start).count() << "sec"
         << std::endl;
    getchar();

    delete[] matrixA;
    delete[] matrixB;
    glDeleteBuffers(1, &matrixBufferA);
    glDeleteBuffers(1, &matrixBufferB);
    glDeleteBuffers(1, &matrixBufferResult);
    glDeleteProgram(computeProgram);
    glDeleteShader(computeShader);
    glfwTerminate();
    return 0;
}