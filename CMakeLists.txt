cmake_minimum_required(VERSION 3.15)
project(fast_thompson LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(fast_thompson src/pybind_module.cpp)

target_include_directories(fast_thompson PRIVATE include)