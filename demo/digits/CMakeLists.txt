
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB SOURCES main.cu)

if (ENABLE_CUDA)
	project("digits_demo" LANGUAGES CXX CUDA)

	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v")
	# -fno-fast-math --fmad=false"

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

	find_package(CUDAToolkit REQUIRED)

	add_executable(digits_demo ${SOURCES})
	target_link_libraries(digits_demo PRIVATE SimpleWindow CUDA::nvml)
else()
	project("digits_demo" LANGUAGES CXX)
	set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)
	add_executable(digits_demo ${SOURCES})
	target_link_libraries(digits_demo PRIVATE SimpleWindow)
endif()

target_include_directories(digits_demo PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../utils" "${stb_SOURCE_DIR}")

add_custom_target(copy_assets_digits ALL
	COMMENT "Copying assets to build directory"
	COMMAND ${CMAKE_COMMAND} -E copy_directory
			${CMAKE_CURRENT_SOURCE_DIR}/data
			${CMAKE_CURRENT_BINARY_DIR}/data
)

add_dependencies(${PROJECT_NAME} copy_assets_digits)
