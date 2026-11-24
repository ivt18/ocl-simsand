#pragma once

#include "cl_setup.h"

#include <cstdint>
#include <string>
#include <vector>

// Owns the cell grid on the GPU and runs the kernels in kernels/simulate.cl.
// Cells are one cl_uint each, row-major: cell (x, y) is at y * width + x.
class Simulation {
public:
    Simulation(const ClSetup& cl, int width, int height, const std::string& kernelDir);

    int width() const { return width_; }
    int height() const { return height_; }

    // Replaces the current state with `cells` (width * height material IDs).
    void upload(const std::vector<cl_uint>& cells);

    // Runs one step: reads the current buffer, writes the other one, then swaps them.
    void step();

    // Converts the current state to RGBA8 pixels (one per cell) and copies them into `pixels`.
    void readPixels(std::vector<uint32_t>& pixels);

private:
    cl::CommandQueue queue_;  // handle copy: cheap, reference-counted
    int width_;
    int height_;
    cl_uint frame_ = 0;

    cl::Program program_;
    cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int, int, cl_uint> step_;      // step(in, out, claimed, width, height, frame)
    cl::KernelFunctor<cl::Buffer, cl::Buffer, int, int> colorize_;  // colorize(cells, pixels, width, height)

    cl::Buffer cells_[2];  // ping-pong pair
    int current_ = 0;      // index of the buffer holding the current state
    cl::Buffer pixels_;
    cl::Buffer claimed_;

    cl::NDRange global_;  // grid size rounded up to whole work-groups
    cl::NDRange local_;   // work-group size
};
