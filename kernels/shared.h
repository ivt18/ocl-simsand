// Constants shared between C++ (src/) and the OpenCL kernels (kernels/).
// Keep this file to #defines and comments only so it stays valid in both languages.
#ifndef SIMSAND_SHARED_H
#define SIMSAND_SHARED_H

// Work-group size for the 2D kernels: WORKGROUP_SIZE x WORKGROUP_SIZE work-items
// (the OpenCL equivalent of threads per block).
#define WORKGROUP_SIZE 16

// Largest 32-bit unsigned value. C++ gets it from <cstdint>, but OpenCL C has no <stdint.h>, so it is
// only defined here for the kernels (__OPENCL_VERSION__ is set by every OpenCL C compiler).
#if defined(__OPENCL_VERSION__) && !defined(UINT32_MAX)
#define UINT32_MAX 0xFFFFFFFFu
#endif

// Color shimmer amount
#define SHIMMER_AMOUNT 0.1f

// Gravity acceleration (16ths of a cell per step)
#define GRAVITY 8

// Maximum velocity a cell can reach in any either axis (16ths of a cell per step)
#define MAX_VELOCITY 112

// Cell masks and offsets
#define MATERIAL_MASK 0xFFu
#define VX_OFFSET 8
#define VY_OFFSET 16
#define FX_OFFSET 24
#define FY_OFFSET 28

// Material types (up to 8 bits, i.e. between 0 and 2^8 - 1)
#define MAT_EMPTY 0u
#define MAT_SAND  1u
#define MAT_STONE 2u

#endif // SIMSAND_SHARED_H
