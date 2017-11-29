cmake_minimum_required(VERSION 2.8)
project(task2)

set(CMAKE_CXX_STANDARD 98)
set(CMAKE_CXX_COMPILER mpic++)
#set(CMAKE_CXX_COMPILER g++)

set(SOURCE_FILES main.cpp)
add_executable(task2 ${SOURCE_FILES})

