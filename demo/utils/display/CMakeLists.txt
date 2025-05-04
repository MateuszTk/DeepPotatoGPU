
cmake_minimum_required(VERSION 3.20)

project(SimpleWindow)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(SimpleWindow STATIC
	"canvas.cpp"
	"canvas.hpp"
)

target_include_directories(SimpleWindow PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if (WIN32)
	target_link_libraries(SimpleWindow PUBLIC user32 gdi32)
else()
	target_link_libraries(SimpleWindow PUBLIC X11)
endif()
