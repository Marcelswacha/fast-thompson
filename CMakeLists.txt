cmake_minimum_required(VERSION 3.15)
project(fast_thompson LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_PYTHON_MODULE "Build the pybind11 Python module" ON)
option(BUILD_BENCHMARK "Build the standalone C++ benchmark" OFF)

if(BUILD_PYTHON_MODULE)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(fast_thompson src/pybind_module.cpp)
    target_include_directories(fast_thompson PRIVATE include)
    target_compile_options(fast_thompson PRIVATE -O3 -mavx2)

    install(TARGETS fast_thompson
            LIBRARY DESTINATION .)
endif()

if(BUILD_BENCHMARK)
    add_executable(ft_bench benchmark/bench.cpp)
    target_include_directories(ft_bench PRIVATE include)
    # -g + frame pointers so `perf record -g` gets clean call stacks
    target_compile_options(ft_bench PRIVATE
        -O3 -mavx2 -g -fno-omit-frame-pointer)
endif()