CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(sumArray)
FIND_PACKAGE(CUDA REQUIRED)
CUDA_ADD_EXECUTABLE(sumArray sumArray.cu)
TARGET_LINK_LIBRARIES(sumArray)