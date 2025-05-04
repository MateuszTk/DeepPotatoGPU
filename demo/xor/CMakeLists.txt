
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES 52)
endif()

if(NOT DEFINED CMAKE_CUDA_COMPILER OR NOT EXISTS ${CMAKE_CUDA_COMPILER})
    set(ENABLE_CUDA FALSE)
else()
    set(ENABLE_CUDA TRUE)
endif()

if (ENABLE_CUDA)
	project("xor_demo" LANGUAGES CXX CUDA)
	add_compile_definitions(CUDA_AVAILIABLE)
else()
	project("xor_demo" LANGUAGES CXX)
endif()

if (ENABLE_CUDA)
	# Enable CUDA debugging if debug build
	if (CMAKE_BUILD_TYPE MATCHES Debug)
		set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
	endif()
endif()

include(CTest)
enable_testing()

if (ENABLE_CUDA)
	# Set include directories for Visual Studio IntelliSense
	# Not required for compilation, as CMake already supports CUDA
	if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
		find_package(CUDA REQUIRED)
		include_directories(${CUDA_INCLUDE_DIRS})
	endif()
endif()

add_executable(xor_demo "main.cu")

target_link_libraries(xor_demo PRIVATE SimpleWindow)
target_include_directories(xor_demo PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../utils")

