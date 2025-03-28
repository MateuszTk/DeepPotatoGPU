
cmake_minimum_required(VERSION 3.20)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
	cmake_policy(SET CMP0141 NEW)
	set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES 52)
endif()

if(NOT DEFINED CMAKE_CUDA_COMPILER OR NOT EXISTS ${CMAKE_CUDA_COMPILER})
    set(ENABLE_CUDA FALSE)
else()
    set(ENABLE_CUDA TRUE)
endif()

if (ENABLE_CUDA)
	project("DeepPotatoGPU" LANGUAGES CXX CUDA)
	add_compile_definitions(CUDA_AVAILIABLE)
else()
	project("DeepPotatoGPU" LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

file(GLOB SOURCES "src/*.cu" "src/*.cpp")
include_directories("src/")

if (NOT ENABLE_CUDA)
	set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)
endif()

add_subdirectory("src/utils/display")

add_executable(DeepPotatoGPU ${SOURCES})

target_link_libraries(DeepPotatoGPU PUBLIC SimpleWindow)

