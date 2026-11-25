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

// Try the two cells diagonally below cell i, in a random order. Returns the target, or -1
int diagonalTarget(__global const uint* in, volatile __global uint* claimed,
                    int x, int y, int i, int width, uint frame)
{
    int below_i = i + width;
    int dir = (hash(x, y, frame) & 1) ? 1 : -1;
    if (x + dir >= 0 && x + dir < width && in[below_i + dir] == MAT_EMPTY && claim(claimed, below_i + dir)) {
        return below_i + dir;
    }
    if (x - dir >= 0 && x - dir < width && in[below_i - dir] == MAT_EMPTY && claim(claimed, below_i - dir)) {
        return below_i - dir;
    }
    return -1;
}

// Simulation step for just the grains of sand that have empty space below
__kernel void step_sand_bottoms(__global const uint* in, __global uint* out,
                                volatile __global uint* claimed, __global uchar* moved,
                                int width, int height, uint frame)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y + 1 >= height)
        return;
    int i = y * width + x;
    int below_i = i + width;
    int target_sand = -1;
    
    if (in[i] != MAT_SAND || in[below_i] == MAT_SAND)
        return;

    // Sand moves straight down if possible. Otherwise, diagonally down
    if (in[below_i] == MAT_EMPTY && claim(claimed, below_i)) {
        target_sand = below_i;
    } else {
        target_sand = diagonalTarget(in, claimed, x, y, i, width, frame);
    }

    if (target_sand >= 0) {
        out[target_sand] = in[i];
        out[i] = MAT_EMPTY;
        moved[i] = 1;
    }
}

__kernel void step_sand_inner(__global const uint* in, __global uint* out,
                              volatile __global uint* claimed, __global uchar* moved,
                              int width, int height, uint frame)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y + 1 >= height)
        return;
    int i = y * width + x;
    int below_i = i + width;

    if (in[i] != MAT_SAND || in[below_i] != MAT_SAND)
        return;

    // Walk down to the bottom grain of this column of sand
    int bottom = below_i;
    while (bottom + width < width * height && in[bottom + width] == MAT_SAND) {
        bottom += width;
    }

    if (moved[bottom]) {
        out[below_i] = MAT_SAND;        // Every grain of sand moves down one cell
        if (y == 0 || in[i - width] != MAT_SAND)
            out[i] = MAT_EMPTY;         // The top grain in the column leaves its cell empty
    } else {
        int target = diagonalTarget(in, claimed, x, y, i, width, frame);
        if (target >= 0) {
            out[target] = in[i];
            out[i] = MAT_EMPTY;
        }
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
