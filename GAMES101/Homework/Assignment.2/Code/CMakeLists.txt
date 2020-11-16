cmake_minimum_required(VERSION 3.10)
project(Rasterizer)

find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 17)

include_directories(/usr/local/include)

add_executable(${PROJECT_NAME} main.cpp rasterizer.hpp rasterizer.cpp global.hpp Triangle.hpp Triangle.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
