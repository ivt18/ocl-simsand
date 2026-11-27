#include "simulation.h"

#include "shared.h"

#include <algorithm>
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
      collide_(program_, "collide"),
      applyImpulses_(program_, "apply_impulses"),
      paint_(program_, "paint"),
      colorize_(program_, "colorize") {
    const size_t gridBytes = size_t(width_) * height_ * sizeof(cl_uint);
    for (cl::Buffer& buffer : cells_) {
        buffer = cl::Buffer(cl.context, CL_MEM_READ_WRITE, gridBytes);
        queue_.enqueueFillBuffer(buffer, MAT_EMPTY, 0, gridBytes);  // contents start out undefined
    }
    pixels_ = cl::Buffer(cl.context, CL_MEM_WRITE_ONLY, gridBytes);
    claimed_ = cl::Buffer(cl.context, CL_MEM_READ_WRITE, gridBytes);

    // Collision buffers. All start out zeroed so they never hold undefined contents.
    const size_t cellCount = size_t(width_) * height_;
    hitGrain_ = cl::Buffer(cl.context, CL_MEM_READ_WRITE, cellCount * sizeof(cl_int));
    hitNormal_ = cl::Buffer(cl.context, CL_MEM_READ_WRITE, cellCount * sizeof(cl_char2));
    dest_ = cl::Buffer(cl.context, CL_MEM_READ_WRITE, cellCount * sizeof(cl_int));
    impulse_ = cl::Buffer(cl.context, CL_MEM_READ_WRITE, cellCount * 2 * sizeof(cl_int));
    queue_.enqueueFillBuffer(hitGrain_, cl_int(-1), 0, cellCount * sizeof(cl_int));
    queue_.enqueueFillBuffer(hitNormal_, cl_char(0), 0, cellCount * sizeof(cl_char2));
    queue_.enqueueFillBuffer(dest_, cl_int(0), 0, cellCount * sizeof(cl_int));
    queue_.enqueueFillBuffer(impulse_, cl_int(0), 0, cellCount * 2 * sizeof(cl_int));

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
    const size_t cellCount = size_t(width_) * height_;
    queue_.enqueueCopyBuffer(cells_[current_], cells_[1 - current_], 0, 0, gridBytes);  // out = in
    queue_.enqueueFillBuffer(claimed_, cl_uint(0), 0, gridBytes);   // clear claimed buffer
    queue_.enqueueFillBuffer(hitGrain_, cl_int(-1), 0, cellCount * sizeof(cl_int));        // no hits yet
    queue_.enqueueFillBuffer(impulse_, cl_int(0), 0, cellCount * 2 * sizeof(cl_int));    // no impulses yet

    // Movement passes: move the grains and record hits in hitGrain/hitNormal and destinations in dest
    stepSandBottoms_(cl::EnqueueArgs(queue_, global_, local_),
            cells_[current_], cells_[1 - current_], claimed_,
            hitGrain_, hitNormal_, dest_, width_, height_, frame_);
    stepSandInner_(cl::EnqueueArgs(queue_, global_, local_),
            cells_[current_], cells_[1 - current_], claimed_,
            hitGrain_, hitNormal_, dest_, width_, height_, frame_);

    // Collision passes: the in-order queue starts each kernel only after the previous one has
    // finished, so collide sees every grain's hit and destination
    collide_(cl::EnqueueArgs(queue_, global_, local_),
            cells_[current_], hitGrain_, hitNormal_, dest_, impulse_, width_, height_);
    applyImpulses_(cl::EnqueueArgs(queue_, global_, local_),
            cells_[current_], cells_[1 - current_], dest_, hitGrain_, hitNormal_, impulse_, width_, height_);
    current_ ^= 1;
    frame_++;
}

void Simulation::paint(int x, int y, int radius, cl_uint material, float density) {
    // The kernel compares against an 8-bit random number, so 256 means "always"
    const cl_uint threshold = cl_uint(std::clamp(density, 0.0f, 1.0f) * 256.0f);
    paint_(cl::EnqueueArgs(queue_, global_, local_),
           cells_[current_], width_, height_, x, y, radius, material, threshold, frame_);
}

void Simulation::readPixels(std::vector<uint32_t>& pixels) {
    colorize_(cl::EnqueueArgs(queue_, global_, local_), cells_[current_], pixels_, width_, height_);

    // Blocking read: waits until all commands before it in the queue have finished.
    pixels.resize(size_t(width_) * height_);
    queue_.enqueueReadBuffer(pixels_, CL_TRUE, 0, pixels.size() * sizeof(uint32_t), pixels.data());
}
