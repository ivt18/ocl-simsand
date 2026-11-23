#include "simulation.h"

#include "shared.h"

#include <stdexcept>

namespace {

size_t roundUp(size_t value, size_t multiple) { return (value + multiple - 1) / multiple * multiple; }

} // namespace

Simulation::Simulation(const ClSetup& cl, int width, int height, const std::string& kernelDir)
    : queue_(cl.queue),
      width_(width),
      height_(height),
      program_(buildProgram(cl, kernelDir + "/simulate.cl", "-I " + kernelDir)),
      step_(program_, "step"),
      colorize_(program_, "colorize") {
    const size_t bytes = size_t(width_) * height_ * sizeof(cl_uint);
    for (cl::Buffer& buffer : cells_) {
        buffer = cl::Buffer(cl.context, CL_MEM_READ_WRITE, bytes);
        queue_.enqueueFillBuffer(buffer, MAT_EMPTY, 0, bytes);  // contents start out undefined
    }
    pixels_ = cl::Buffer(cl.context, CL_MEM_WRITE_ONLY, bytes);

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
    // ≈ step<<<global / local, local>>>(in, out, width, height)
    step_(cl::EnqueueArgs(queue_, global_, local_), cells_[current_], cells_[1 - current_], width_, height_);
    current_ ^= 1;
}

void Simulation::readPixels(std::vector<uint32_t>& pixels) {
    colorize_(cl::EnqueueArgs(queue_, global_, local_), cells_[current_], pixels_, width_, height_);

    // Blocking read: waits until all commands before it in the queue have finished.
    pixels.resize(size_t(width_) * height_);
    queue_.enqueueReadBuffer(pixels_, CL_TRUE, 0, pixels.size() * sizeof(uint32_t), pixels.data());
}
