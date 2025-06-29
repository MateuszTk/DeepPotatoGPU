
cmake_minimum_required(VERSION 3.26)
include(CheckLanguage)

check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
	project("DeepPotatoGPU" LANGUAGES CXX CUDA)
else()
	project("DeepPotatoGPU" LANGUAGES CXX)
endif()

include_directories("src")
add_subdirectory("demo")



