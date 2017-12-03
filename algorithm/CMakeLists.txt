cmake_minimum_required(VERSION 2.8)
project(task2)

set(CMAKE_CXX_STANDARD 98)
set(CMAKE_CXX_COMPILER mpic++)
include_directories(/usr/include/mpi)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wextra -fopenmp")

set(SOURCE_FILES main.cpp)
add_executable(task2 ${SOURCE_FILES})
