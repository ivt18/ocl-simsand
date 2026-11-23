#include "cl_setup.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

ClSetup createClSetup() {
    std::vector<cl::Platform> platforms;
    try {
        cl::Platform::get(&platforms);
    } catch (const cl::Error&) {
        // No platform at all: handled below.
    }
    if (platforms.empty()) {
        throw std::runtime_error("No OpenCL platforms found. Is a GPU driver with OpenCL support installed?");
    }

    // Prefer a GPU on any platform; otherwise take any device (e.g. a CPU OpenCL driver).
    for (cl_device_type type : {cl_device_type{CL_DEVICE_TYPE_GPU}, cl_device_type{CL_DEVICE_TYPE_ALL}}) {
        for (const cl::Platform& platform : platforms) {
            std::vector<cl::Device> devices;
            try {
                platform.getDevices(type, &devices);
            } catch (const cl::Error&) {
                continue;  // this platform has no device of this type
            }
            if (devices.empty()) {
                continue;
            }

            ClSetup cl;
            cl.device = devices.front();
            cl.context = cl::Context(cl.device);
            cl.queue = cl::CommandQueue(cl.context, cl.device);
            std::printf("OpenCL device: %s (%s)\n", cl.device.getInfo<CL_DEVICE_NAME>().c_str(),
                        cl.device.getInfo<CL_DEVICE_VERSION>().c_str());
            return cl;
        }
    }
    throw std::runtime_error("No OpenCL devices found.");
}

cl::Program buildProgram(const ClSetup& cl, const std::string& path, const std::string& options) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open kernel source: " + path);
    }
    std::stringstream source;
    source << file.rdbuf();

    cl::Program program(cl.context, source.str());
    try {
        program.build({cl.device}, options.c_str());
    } catch (const cl::Error&) {
        const std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl.device);
        throw std::runtime_error("Failed to compile " + path + ":\n" + log);
    }
    return program;
}
