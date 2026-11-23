#pragma once

#include <CL/opencl.hpp>

#include <string>

// The OpenCL objects everything else needs.
struct ClSetup {
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;  // in-order: commands run one after another, like a CUDA stream
};

// Picks the first GPU from any OpenCL platform (falling back to any device),
// then creates a context and a command queue for it.
ClSetup createClSetup();

// Compiles an OpenCL C source file for the device. `options` go to the OpenCL compiler,
// e.g. "-I <dir>". Throws std::runtime_error with the compiler's log if compilation fails.
cl::Program buildProgram(const ClSetup& cl, const std::string& path, const std::string& options);
