# CMakeList.txt: проект CMake для CUDA_CMake; включите исходный код и определения,
# укажите здесь логику для конкретного проекта.
#
cmake_minimum_required (VERSION 3.8)

# Включение горячей перезагрузки для компиляторов MSVC, если поддерживается.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("CUDA_CMake")

enable_language(CUDA)
include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/include")

# Добавьте источник в исполняемый файл этого проекта.
add_executable (LAB02_GPU "LR02/src/LR02_2G.cu")
add_executable (LAB02_CPU "LR02/src/LR02_2C.cpp")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET LAB02_CPU PROPERTY CXX_STANDARD 20)
  set_property(TARGET LAB02_GPU PROPERTY CUDA_ARCHITECTURES 75)
endif()

# TODO: Добавьте тесты и целевые объекты, если это необходимо.
