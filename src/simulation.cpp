#include "simulation.h"

#include "shared.h"

#include <stdexcept>

namespace {

size_t roundUp(size_t value, size_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
}

} // namespace

Simulation::Simulation(const ClSetup& cl, int width, int height, const std::string& kernelDir)
    : queue_(cl.queue),
      width_(width),
      height_(height),
      program_(buildProgram(cl, kernelDir + "/simulate.cl", "-I " + kernelDir)),
      stepSandBottoms_(program_, "step_sand_bottoms"),
      stepSandInner_(program_, "step_sand_inner"),
      colorize_(program_, "colorize") {
    const size_t gridBytes = size_t(width_) * height_ * sizeof(cl_uint);
    for (cl::Buffer& buffer : cells_) {
        buffer = cl::Buffer(cl.context, CL_MEM_READ_WRITE, gridBytes);
        queue_.enqueueFillBuffer(buffer, MAT_EMPTY, 0, gridBytes);  // contents start out undefined
    }
    pixels_ = cl::Buffer(cl.context, CL_MEM_WRITE_ONLY, gridBytes);
    claimed_ = cl::Buffer(cl.context, CL_MEM_READ_WRITE, gridBytes);

    const size_t movedBytes = size_t(width_) * height_ * sizeof(cl_uchar);
    moved_ = cl::Buffer(cl.context, CL_MEM_READ_WRITE, movedBytes);
    queue_.enqueueFillBuffer(moved_, cl_uchar(0), 0, movedBytes);  // contents start out undefined

    // Like a CUDA launch: global = total work-items (rounded up), local = work-group (block) size.
    global_ = cl::NDRange(roundUp(width_, WORKGROUP_SIZE), roundUp(height_, WORKGROUP_SIZE));
    local_ = cl::NDRange(WORKGROUP_SIZE, WORKGROUP_SIZE);
}

void Simulation::upload(const std::vector<cl_uint>& cells) {
    if (cells.size() != size_t(width_) * height_) {
        throw std::invalid_argument("Simulation::upload: expected width * height cells");
    }
    queue_.enqueueWriteBuffer(cells_[current_], CL_TRUE, 0, cells.size() * sizeof(cl_uint), cells.data());
}

void Simulation::step() {
    const size_t gridBytes = size_t(width_) * size_t(height_) * sizeof(cl_uint);
    const size_t movedBytes = size_t(width_) * height_ * sizeof(cl_uchar);
    queue_.enqueueCopyBuffer(cells_[current_], cells_[1 - current_], 0, 0, gridBytes);  // out = in
    queue_.enqueueFillBuffer(claimed_, cl_uint(0), 0, gridBytes);   // clear claimed buffer
    queue_.enqueueFillBuffer(moved_, cl_uchar(0), 0, movedBytes);   // clear moved buffer
    stepSandBottoms_(cl::EnqueueArgs(queue_, global_, local_),
            cells_[current_], cells_[1 - current_], claimed_, moved_, width_, height_, frame_);
    stepSandInner_(cl::EnqueueArgs(queue_, global_, local_),
            cells_[current_], cells_[1 - current_], claimed_, moved_, width_, height_, frame_);
    current_ ^= 1;
    frame_++;
}

void Simulation::readPixels(std::vector<uint32_t>& pixels) {
    colorize_(cl::EnqueueArgs(queue_, global_, local_), cells_[current_], pixels_, width_, height_);

    // Blocking read: waits until all commands before it in the queue have finished.
    pixels.resize(size_t(width_) * height_);
    queue_.enqueueReadBuffer(pixels_, CL_TRUE, 0, pixels.size() * sizeof(uint32_t), pixels.data());
}
