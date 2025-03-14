
cmake_minimum_required(VERSION 3.20)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
	cmake_policy(SET CMP0141 NEW)
	set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES 52)
endif()

project("DeepPotatoGPU" LANGUAGES CXX CUDA)

# Enable CUDA debugging if debug build
if (CMAKE_BUILD_TYPE MATCHES Debug)
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
endif()

include(CTest)
enable_testing()

# Set include directories for Visual Studio IntelliSense
# Not required for compilation, as CMake already supports CUDA
if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
	find_package(CUDA REQUIRED)
	include_directories(${CUDA_INCLUDE_DIRS})
endif()

file(GLOB SOURCES "src/*.cu" "src/*.cpp")

add_executable(DeepPotatoGPU ${SOURCES})

set_property(TARGET DeepPotatoGPU PROPERTY CXX_STANDARD 20)
