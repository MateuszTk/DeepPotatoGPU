
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

set(FETCHCONTENT_QUIET OFF)
include(FetchContent)

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

FetchContent_Declare(
	stb
	GIT_REPOSITORY https://github.com/nothings/stb
	GIT_TAG 5736b15f7ea0ffb08dd38af21067c314d6a3aae9
	GIT_PROGRESS TRUE
)

FetchContent_MakeAvailable(stb)

file(GLOB SOURCES "src/*.cu" "src/*.cpp")
include_directories("src/" ${stb_SOURCE_DIR})

if (NOT ENABLE_CUDA)
	set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)
endif()

add_subdirectory("src/utils/display")

add_executable(DeepPotatoGPU ${SOURCES})

target_link_libraries(DeepPotatoGPU PUBLIC SimpleWindow)

add_custom_target(copy_assets ALL
	COMMENT "Copying assets to build directory"
	COMMAND ${CMAKE_COMMAND} -E copy_directory
			${CMAKE_CURRENT_SOURCE_DIR}/data
			${CMAKE_CURRENT_BINARY_DIR}/data
)

add_dependencies(${PROJECT_NAME} copy_assets)

