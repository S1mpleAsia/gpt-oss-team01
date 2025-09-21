cmake_minimum_required(VERSION 3.21)
project(gemm_bench LANGUAGES CXX HIP)

find_package(HIP REQUIRED)

set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

option(WITH_ROCBLAS "Link against rocBLAS for reference timing" ON)

if(WITH_ROCBLAS)
  find_package(rocblas REQUIRED)
endif()

set_source_files_properties(gemm_bench.cpp PROPERTIES LANGUAGE HIP)

hip_add_executable(gemm_bench gemm_bench.cpp)

if(WITH_ROCBLAS)
  target_compile_definitions(gemm_bench PRIVATE WITH_ROCBLAS=1)
  target_link_libraries(gemm_bench PRIVATE roc::rocblas)
else()
  target_compile_definitions(gemm_bench PRIVATE WITH_ROCBLAS=0)
endif()

target_include_directories(gemm_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
