# CMakeList.txt: проект CMake для CUDA_CMake; включите исходный код и определения,
# укажите здесь логику для конкретного проекта.
#
cmake_minimum_required (VERSION 3.8)

# Включение горячей перезагрузки для компиляторов MSVC, если поддерживается.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

set (CMAKE_CXX_STANDARD 17)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

project ("CUDA_CMake")
include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/include")
enable_language(CUDA)

# OpenGL, GLFW и GLEW свистопляски
add_subdirectory(glfw glfw/src)
find_package(OpenGL 32 COMPONENTS OpenGL)
include_directories(${OPENGL_INCLUDE_DIRS} ${GLEW_LIBRARIES})
find_package(GLEW REQUIRED GLEW::glew_s)
#set(GL_LIBRARY GL GLU X11)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# Добавьте источник в исполняемый файл этого проекта.
add_executable (LR02_GPU "LR02/src/LR02_1G.cu")
add_executable (LR02_CPU_SINGLE "LR02/src/LR02_1C.cpp")
add_executable (LR02_CPU_THREADS "LR02/src/LR02_2C.cpp")
add_executable (LR03_GPU "LR03/src/LR03_1G.cu")
add_executable (LR03_GPU_2 "LR03/src/LR03_2G.cu")
add_executable (LR04_GPU "LR04/src/LR04_1G.cu")
add_executable (LR05_GPU "LR05/src/LR05_1G.cu")

add_executable (LR07_GPU_1 "LR07/src/LR07_1G.cu")
add_executable (LR07_GPU_2 "LR07/src/LR07_2G.cu")

add_executable (LR08_cuBLAS "LR08/src/LR08_1G.cu")
add_executable (LR08_CUDA "LR08/src/LR08_2G.cu")

add_executable (LR09_COPY "LR09/src/LR09_1G.cu")
add_executable (LR09_STREAM "LR09/src/LR09_2G.cu")
add_executable (LR09_SCALAR "LR09/src/LR09_3G.cu")

#add_executable (LR10_PY "LR10/src/LR10_1G.py")
add_executable (LR10_cuBLAS "LR10/src/LR10_1G.cu")

add_executable (
        RGZ_MatrixMultiplication "RGZ/src/main.cpp"
        RGZ/src/RGZ.h
        RGZ/src/Matrix.cpp
        RGZ/src/Opengl.cpp
)
add_dependencies(RGZ_MatrixMultiplication OpenGL::GL GLEW::glew)
target_link_libraries(RGZ_MatrixMultiplication ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} glfw)

# CUDA_ARCHITECTURES 35;37;50;52;53;60;61;62;70;72;75;80;86;87
if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET LR02_CPU_THREADS PROPERTY CXX_STANDARD 20)
  set_property(TARGET LR02_CPU_SINGLE PROPERTY CXX_STANDARD 20)
  set_property(TARGET LR02_GPU PROPERTY CUDA_ARCHITECTURES 62)
  set_property(TARGET LR03_GPU PROPERTY CUDA_ARCHITECTURES 62)
  set_property(TARGET LR03_GPU_2 PROPERTY CUDA_ARCHITECTURES 52)
  set_property(TARGET LR04_GPU PROPERTY CUDA_ARCHITECTURES 52)
  set_property(TARGET LR05_GPU PROPERTY CUDA_ARCHITECTURES 61)

  set_property(TARGET LR07_GPU_1 PROPERTY CUDA_ARCHITECTURES 61)
  set_property(TARGET LR07_GPU_2 PROPERTY CUDA_ARCHITECTURES 61)

  set_property(TARGET LR08_CUDA PROPERTY CUDA_ARCHITECTURES 61)
  set_property(TARGET LR08_cuBLAS PROPERTY CUDA_ARCHITECTURES 61)

  set_property(TARGET LR09_COPY PROPERTY CUDA_ARCHITECTURES 61)
  set_property(TARGET LR09_STREAM PROPERTY CUDA_ARCHITECTURES 61)
  set_property(TARGET LR09_SCALAR PROPERTY CUDA_ARCHITECTURES 61)

  set_property(TARGET LR10_cuBLAS PROPERTY CUDA_ARCHITECTURES 61)

endif()