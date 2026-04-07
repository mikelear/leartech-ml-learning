cmake_minimum_required(VERSION 3.18)
project(ml_learning_session6 CXX)

set(CMAKE_CXX_STANDARD 17)
# -O0 disables optimisation so the debugger shows every step
# -g enables debug symbols
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")

add_executable(loss loss.cpp)
