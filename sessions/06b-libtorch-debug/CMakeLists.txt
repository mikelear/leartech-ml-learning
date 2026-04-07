cmake_minimum_required(VERSION 3.18)
project(libtorch_debug CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")

# Find libtorch in the local directory
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch")
find_package(Torch REQUIRED)

add_executable(libtorch_debug main.cpp)
target_link_libraries(libtorch_debug "${TORCH_LIBRARIES}")
