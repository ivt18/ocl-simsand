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
    // All step kernels take (in, out, claimed, hitGrain, hitNormal, dest, width, height, frame).
    using StepKernel = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer,
                                         cl::Buffer, cl::Buffer, cl::Buffer, int, int, cl_uint>;
    StepKernel stepSandBottoms_;  // step_sand_bottoms
    StepKernel stepSandInner_;    // step_sand_inner
    // collide(in, hitGrain, hitNormal, dest, impulse, width, height)
    cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int, int> collide_;
    // apply_impulses(in, out, dest, hitGrain, hitNormal, impulse, width, height)
    cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int, int>
        applyImpulses_;
    cl::KernelFunctor<cl::Buffer, cl::Buffer, int, int> colorize_;  // colorize(cells, pixels, width, height)

    cl::Buffer cells_[2];  // ping-pong pair
    int current_ = 0;      // index of the buffer holding the current state
    cl::Buffer pixels_;
    cl::Buffer claimed_;
    // Collision buffers, all indexed by a grain's start-of-step index:
    cl::Buffer hitGrain_;   // one cl_int per cell: index of the grain this one hit (-1 = none); cleared to -1 every step
    cl::Buffer hitNormal_;  // one cl_char2 per cell: the blocked step; only valid where hitGrain >= 0
    cl::Buffer dest_;       // one cl_int per cell: index of the cell the grain ended up in; every grain writes it
    cl::Buffer impulse_;    // two cl_ints per cell (x, y): summed velocity changes; cleared every step

    cl::NDRange global_;  // grid size rounded up to whole work-groups
    cl::NDRange local_;   // work-group size
};
