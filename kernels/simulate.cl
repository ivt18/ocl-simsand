#include "shared.h"

// One simulation step, one work-item per cell.
// Ping-pong: reads the current state from `in` and writes the next state to `out`.
// The host swaps the two buffers after every step.
__kernel void step(__global const uint* in, __global uint* out, int width, int height)
{
    // The global size is rounded up to whole work-groups, so some work-items fall
    // outside the grid. They have no cell; exit before reading or writing out of bounds.
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    int i = y * width + x;

    // PLACEHOLDER: copies the state unchanged. Replace with the sand rules.
    // Every in-bounds cell of `out` must be written on every step; any cell that
    // isn't keeps its value from two steps ago.
    out[i] = in[i];
}

// Material colours (RGBA, sRGB, 0..1), indexed by material ID (MAT_* in shared.h).
// float4 rather than float3: arrays of float3 are laid out inconsistently between
// compilers (NVIDIA packs them at 12 bytes but indexes at 16), which scrambles the colours.
__constant float4 MATERIAL_COLORS[] = {
    (float4)(0.06f, 0.07f, 0.09f, 1.0f),  // MAT_EMPTY
    (float4)(0.95f, 0.82f, 0.50f, 1.0f),  // MAT_SAND
    (float4)(0.58f, 0.58f, 0.58f, 1.0f),  // MAT_STONE
};
#define MATERIAL_COUNT (sizeof(MATERIAL_COLORS) / sizeof(MATERIAL_COLORS[0]))

// Turns cells into pixels for display: one RGBA8 pixel per cell.
__kernel void colorize(__global const uint* cells, __global uint* pixels, int width, int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    int i = y * width + x;

    uint material = cells[i];
    float3 color = material < MATERIAL_COUNT ? MATERIAL_COLORS[material].xyz
                                             : (float3)(1.0f, 0.0f, 1.0f);  // magenta = unknown ID

    // Byte order R, G, B, A in memory (SDL_PIXELFORMAT_RGBA32), so R is the lowest byte.
    uint3 rgb = convert_uint3_sat_rte(color * 255.0f);
    pixels[i] = rgb.x | (rgb.y << 8) | (rgb.z << 16) | (0xFFu << 24);
}
