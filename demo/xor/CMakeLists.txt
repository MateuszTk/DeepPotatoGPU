
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB SOURCES main.cu)

if (ENABLE_CUDA)
	project("xor_demo" LANGUAGES CXX CUDA)

	# Enable CUDA debugging if debug build
	if (CMAKE_BUILD_TYPE MATCHES Debug)
		set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
	endif()

	# Set include directories for Visual Studio IntelliSense
	# Not required for compilation, as CMake already supports CUDA
	if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
		find_package(CUDA REQUIRED)
		include_directories(${CUDA_INCLUDE_DIRS})
	endif()
else()
	project("xor_demo" LANGUAGES CXX)
	set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)
endif()

add_executable(xor_demo ${SOURCES})

target_link_libraries(xor_demo PRIVATE SimpleWindow)
target_include_directories(xor_demo PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../utils")

