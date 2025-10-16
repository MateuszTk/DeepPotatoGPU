
cmake_minimum_required(VERSION 3.26)

option(ENABLE_WMMA "Use Tensor Cores via WMMA API" OFF)
option(ENABLE_FP16 "Use FP16 data type" OFF)
option(DISABLE_CUDA "Disable CUDA even if available" OFF)

include(CheckLanguage)
check_language(CUDA)

if (CMAKE_CUDA_COMPILER AND NOT DISABLE_CUDA)
	set(CMAKE_CUDA_ARCHITECTURES 75)
	project("DeepPotatoGPU" LANGUAGES CXX CUDA)
	set(ENABLE_CUDA TRUE)

	add_compile_definitions(CUDA_AVAILABLE)
	if (ENABLE_WMMA)
		add_compile_definitions(ENABLE_WMMA)
	endif()
	if (ENABLE_FP16)
		add_compile_definitions(ENABLE_FP16)
	endif()
else()
	project("DeepPotatoGPU" LANGUAGES CXX)
	set(ENABLE_CUDA FALSE)
endif()

include_directories("src")
add_subdirectory("demo")



