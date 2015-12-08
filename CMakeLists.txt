cmake_minimum_required(VERSION 3.3)
project(Jacobi_omp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

set(SOURCE_FILES main.cpp)
#set(SOURCE_FILES segv.cpp)
add_executable(Jacobi_omp ${SOURCE_FILES})
#add_executable(segv ${SOURCE_FILES})