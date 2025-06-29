
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

if(NOT DEFINED CMAKE_CUDA_COMPILER OR NOT EXISTS ${CMAKE_CUDA_COMPILER})
    set(ENABLE_CUDA FALSE)
else()
    set(ENABLE_CUDA TRUE)
endif()

if (ENABLE_CUDA)
	project("compression_demo" LANGUAGES CXX CUDA)
	add_compile_definitions(CUDA_AVAILIABLE)
else()
	project("compression_demo" LANGUAGES CXX)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v")

if (ENABLE_CUDA)
	# Enable CUDA debugging if debug build
	if (CMAKE_BUILD_TYPE MATCHES Debug)
		set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
	endif()
endif()

include(CTest)
enable_testing()

set(FETCHCONTENT_QUIET OFF)
include(FetchContent)

FetchContent_Declare(
	stb
	GIT_REPOSITORY https://github.com/nothings/stb
	GIT_TAG 5736b15f7ea0ffb08dd38af21067c314d6a3aae9
	GIT_PROGRESS TRUE
)

FetchContent_MakeAvailable(stb)

if (ENABLE_CUDA)
	# Set include directories for Visual Studio IntelliSense
	# Not required for compilation, as CMake already supports CUDA
	if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
		find_package(CUDA REQUIRED)
		include_directories(${CUDA_INCLUDE_DIRS})
	endif()
endif()

add_executable(compression_demo "main.cu")

target_link_libraries(compression_demo PRIVATE SimpleWindow)
target_include_directories(compression_demo PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../utils" "${stb_SOURCE_DIR}")

add_custom_target(copy_assets ALL
	COMMENT "Copying assets to build directory"
	COMMAND ${CMAKE_COMMAND} -E copy_directory
			${CMAKE_CURRENT_SOURCE_DIR}/data
			${CMAKE_CURRENT_BINARY_DIR}/data
)

add_dependencies(${PROJECT_NAME} copy_assets)
