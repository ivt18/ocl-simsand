cmake_minimum_required(VERSION 3.24)
project(ocl-simsand LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# SDL3: window, input and drawing. Built as a static library so no SDL3.dll is needed next to the .exe.
set(SDL_SHARED       OFF CACHE BOOL "" FORCE)
set(SDL_STATIC       ON  CACHE BOOL "" FORCE)
set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
FetchContent_Declare(SDL3
    GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
    GIT_TAG release-3.4.18
    GIT_SHALLOW TRUE)

# OpenCL: C headers, the ICD loader (OpenCL.lib) and the C++ bindings (CL/opencl.hpp).
# The OpenCL runtime itself comes with the GPU driver.
set(BUILD_TESTING                    OFF CACHE BOOL "" FORCE)
set(OPENCL_ICD_LOADER_BUILD_TESTING  OFF CACHE BOOL "" FORCE)
set(OPENCL_CLHPP_BUILD_TESTING       OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES                   OFF CACHE BOOL "" FORCE)
set(BUILD_DOCS                       OFF CACHE BOOL "" FORCE)
FetchContent_Declare(OpenCLHeaders
    GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-Headers.git
    GIT_TAG v2026.05.29
    GIT_SHALLOW TRUE)
FetchContent_Declare(OpenCLICDLoader
    GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-ICD-Loader.git
    GIT_TAG v2026.05.29
    GIT_SHALLOW TRUE)
FetchContent_Declare(OpenCLHeadersCpp
    GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-CLHPP.git
    GIT_TAG v2026.05.29
    GIT_SHALLOW TRUE)

FetchContent_MakeAvailable(SDL3 OpenCLHeaders OpenCLICDLoader OpenCLHeadersCpp)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
add_executable(ocl-simsand
    src/main.cpp
    src/cl_setup.cpp
    src/simulation.cpp
)
# Listed so they show up in IDEs; kernels are compiled at runtime, not by CMake.
target_sources(ocl-simsand PRIVATE kernels/simulate.cl kernels/shared.h)
set_source_files_properties(kernels/simulate.cl PROPERTIES HEADER_FILE_ONLY TRUE)

target_include_directories(ocl-simsand PRIVATE src kernels)  # kernels/ holds shared.h (C++/OpenCL constants)
target_link_libraries(ocl-simsand PRIVATE SDL3::SDL3 OpenCL::OpenCL OpenCL::HeadersCpp)
target_compile_definitions(ocl-simsand PRIVATE
    CL_TARGET_OPENCL_VERSION=120          # OpenCL 1.2 API: supported by NVIDIA, AMD and Intel
    CL_HPP_TARGET_OPENCL_VERSION=120
    CL_HPP_MINIMUM_OPENCL_VERSION=120
    CL_HPP_ENABLE_EXCEPTIONS              # OpenCL errors throw cl::Error instead of returning codes
    SIMSAND_KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels"
)

if(MSVC)
    target_compile_options(ocl-simsand PRIVATE /W4 /permissive- /utf-8)
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ocl-simsand)
else()
    target_compile_options(ocl-simsand PRIVATE -Wall -Wextra)
endif()
