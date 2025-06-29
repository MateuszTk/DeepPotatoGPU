
cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED CMAKE_CUDA_COMPILER OR NOT EXISTS ${CMAKE_CUDA_COMPILER})
    set(ENABLE_CUDA FALSE)
else()
    set(ENABLE_CUDA TRUE)
endif()

if (ENABLE_CUDA)
	project("DeepPotatoGPU" LANGUAGES CXX CUDA)
else()
	project("DeepPotatoGPU" LANGUAGES CXX)
endif()

include_directories("src")
add_subdirectory("demo")



