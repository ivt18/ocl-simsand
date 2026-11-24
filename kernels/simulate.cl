#include "shared.h"

// Claim cell i for this simulation step. Each cell can be accessed by at most one caller per
// simulation step. Returns true if the cell was claimed successfully, false otherwise
bool claim(volatile __global uint* claimed, int i)
{
    return atomic_cmpxchg(&claimed[i], 0u, 1u) == 0u;
}

// Cheap per-cell and per-frame hash function
uint hash(uint x, uint y, uint frame)
{
    uint h = (x * 0x8da6b343u) ^ (y * 0xd8163841u) ^ (frame * 0xcb1ab31fu);
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    return h;
}

// One simulation step, one work-item per cell.
// Ping-pong: reads the current state from `in` and writes the next state to `out`.
// The host swaps the two buffers after every step.
__kernel void step(__global const uint* in, __global uint* out,
                    volatile __global uint* claimed, int width, int height, uint frame)
{
    // The global size is rounded up to whole work-groups, so some work-items fall
    // outside the grid. They have no cell; exit before reading or writing out of bounds.
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    int i = y * width + x;
    int below_i = i + width;
    int target_sand = -1;

    if (in[i] != MAT_SAND || (in[i] == MAT_SAND && y + 1 >= height)) {
        return;
    }

    // Material rules (only moving materials)
    // Out is initialized as a copy of in, so that static particles stay in place
    switch (in[i]) {
        // Sand moves straight down if possible. Otherwise, diagonally down
        case MAT_SAND: {
            int dir = (hash(x, y, frame) & 1) ? 1 : -1; // Which diagonal to try first
            if (in[below_i] == MAT_EMPTY && claim(claimed, below_i)) {
                target_sand = below_i;
            } else if (x + dir >= 0 && x + dir < width
                && in[below_i + dir] == MAT_EMPTY && claim(claimed, below_i + dir)) {
                target_sand = below_i + dir;
            } else if (x - dir >= 0 && x - dir < width
                && in[below_i - dir] == MAT_EMPTY && claim(claimed, below_i - dir)) {
                target_sand = below_i - dir;
            }
            
            if (target_sand >= 0) {
                out[target_sand] = in[i];
                out[i] = MAT_EMPTY;
            }
            break;
        }

        default:
            break;
    }
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
