// Constants shared between C++ (src/) and the OpenCL kernels (kernels/).
// Keep this file to #defines and comments only so it stays valid in both languages.
#ifndef SIMSAND_SHARED_H
#define SIMSAND_SHARED_H

// Work-group size for the 2D kernels: WORKGROUP_SIZE x WORKGROUP_SIZE work-items
// (the OpenCL equivalent of threads per block).
#define WORKGROUP_SIZE 16

// Material types
#define MAT_EMPTY 0u
#define MAT_SAND  1u
#define MAT_STONE 2u

#endif // SIMSAND_SHARED_H
