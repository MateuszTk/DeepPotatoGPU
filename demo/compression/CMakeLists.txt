
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB SOURCES main.cu)

if (ENABLE_CUDA)
	project("compression_demo" LANGUAGES CXX CUDA)

	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v")

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
	project("compression_demo" LANGUAGES CXX)
	set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)
endif()


set(FETCHCONTENT_QUIET OFF)
include(FetchContent)

FetchContent_Declare(
	stb
	GIT_REPOSITORY https://github.com/nothings/stb
	GIT_TAG 5736b15f7ea0ffb08dd38af21067c314d6a3aae9
	GIT_PROGRESS TRUE
)

FetchContent_MakeAvailable(stb)

add_executable(compression_demo ${SOURCES})

target_link_libraries(compression_demo PRIVATE SimpleWindow)
target_include_directories(compression_demo PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../utils" "${stb_SOURCE_DIR}")

add_custom_target(copy_assets ALL
	COMMENT "Copying assets to build directory"
	COMMAND ${CMAKE_COMMAND} -E copy_directory
			${CMAKE_CURRENT_SOURCE_DIR}/data
			${CMAKE_CURRENT_BINARY_DIR}/data
)

add_dependencies(${PROJECT_NAME} copy_assets)
