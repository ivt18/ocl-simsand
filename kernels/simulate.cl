#include "shared.h"

typedef struct {
    int2 end;          // last cell reached (== start if the grain couldn't move at all)
    int2 blockedStep;  // the step that failed, or (0, 0) if the whole path was claimed
} PathResult;

// Tries to claim 'cell' for the grain that started this step at index 'claimer'.
// Returns 0 on success, otherwise the index of the grain that climed it first, plus 1.
uint tryClaim(volatile __global uint* claimed, int cell, int claimer)
{
    return atomic_cmpxchg(&claimed[cell], 0u, (uint)claimer + 1u);
}

// Claim cell i for this simulation step. Each cell can be accessed by at most one caller per
// simulation step. Returns true if the cell was claimed successfully, false otherwise
bool claim(volatile __global uint* claimed, int cell, int claimer)
{
    return tryClaim(claimed, cell, claimer) == 0u;
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

// Get the material ID of a cell
uint getCellMaterial(uint cell)
{
    return cell & MATERIAL_MASK;  // material lives in the low 8 bits
}

// Get the velocity vector of a cell
int2 getCellVelocity(uint cell)
{
    return (int2)(as_char((uchar)(cell >> VX_OFFSET)),
                  as_char((uchar)(cell >> VY_OFFSET)));
}

// Returns the velocity with each component limited to [-MAX_VELOCITY, MAX_VELOCITY]
int2 clampCellVelocity(int2 velocity)
{
    return clamp(velocity, -MAX_VELOCITY, MAX_VELOCITY);
}

// Get the fractional position of a cell within its current cell, in 16ths of a cell (0..15 per axis)
int2 getCellFractionalMovement(uint cell)
{
    return (int2)((int)((cell >> FX_OFFSET) & 0xFu),
                  (int)((cell >> FY_OFFSET) & 0xFu));
}

// Packs a material, a velocity (each component -128..127) and a fractional position (each
// component 0..15) into one cell. The reverse of getCellMaterial, getCellVelocity and
// getCellFractionalMovement.
uint makeCell(uint material, int2 velocity, int2 frac)
{
    // Masking a negative int keeps its low 8 bits, i.e. its two's complement byte,
    // which getCellVelocity sign-extends back to the same value
    return (material & MATERIAL_MASK)
         | (((uint)velocity.x & 0xFFu) << VX_OFFSET)
         | (((uint)velocity.y & 0xFFu) << VY_OFFSET)
         | (((uint)frac.x & 0xFu) << FX_OFFSET)
         | (((uint)frac.y & 0xFu) << FY_OFFSET);
}

// True if position p lies inside the grid
bool isInGrid(int2 p, int width, int height)
{
    return p.x >= 0 && p.x < width && p.y >= 0 && p.y < height;
}

// Walks a line from 'start' to 'start + delta' one cell at a time, and claims each cell.
// Stops at the first cell that is outside the grid, claimed by another grain, or static.
// Returns the end point of the claimed path. Implemented according to Bresenham's algo.
// Also returns the step that was blocked, so the caller can tell which way the grain hit something.
PathResult claimPath(__global const uint* in, volatile __global uint* claimed,
                     int2 start, int2 delta, int width, int height)
{
    int dx = abs(delta.x);
    int dy = abs(delta.y);
    int2 dir = clamp(delta, -1, 1); // movement direction on each axis
    int err = dx - dy;
    int self = start.y * width + start.x;  // this grain's ID: its index at the start of the step

    PathResult result;
    result.end = start;
    result.blockedStep = (int2)(0, 0);

    int2 p = start;
    for (int k = 0; k < max(dx, dy); k++) {
        // Step along x, along y, or both (a diagonal step), whichever keeps closest to the line
        int e2 = 2 * err;
        int2 step = (int2)(0, 0);
        if (e2 > -dy) {
            err -= dy;
            step.x = dir.x;
        }
        if (e2 < dx) {
            err += dx;
            step.y = dir.y;
        }

        int2 next = p + step;
        int n = next.y * width + next.x;
        if (!isInGrid(next, width, height)
            || getCellMaterial(in[n]) != MAT_EMPTY || !claim(claimed, n, self)) {
            result.blockedStep = step;
            break;
        }
        p = next;
        result.end = p;
    }
    return result;
}

// Try the two cells diagonally below cell i, in a random order. Returns the target, or -1
int diagonalTarget(__global const uint* in, volatile __global uint* claimed,
                    int x, int y, int i, int width, int height, uint frame)
{
    int below_i = i + width;
    int dir = (hash(x, y, frame) & 1) ? 1 : -1;
    if (isInGrid((int2)(x + dir, y + 1), width, height)
        && getCellMaterial(in[below_i + dir]) == MAT_EMPTY && claim(claimed, below_i + dir, i)) {
        return below_i + dir;
    }
    if (isInGrid((int2)(x - dir, y + 1), width, height)
        && getCellMaterial(in[below_i - dir]) == MAT_EMPTY && claim(claimed, below_i - dir, i)) {
        return below_i - dir;
    }
    return -1;
}

// Simulation step for just the grains of sand that don't have another grain directly below them
// (the bottom grains of each column). The grains above them are handled by step_sand_inner.
__kernel void step_sand_bottoms(__global const uint* in, __global uint* out,
                                volatile __global uint* claimed,
                                __global int* hitGrain, __global char2* hitNormal,
                                __global int* dest, int width, int height, uint frame)
{
    int2 pos = (int2)((int)get_global_id(0), (int)get_global_id(1));
    if (!isInGrid(pos, width, height))
        return;

    int i = pos.y * width + pos.x;
    if (getCellMaterial(in[i]) != MAT_SAND)
        return;
    if (pos.y + 1 < height && getCellMaterial(in[i + width]) == MAT_SAND)
        return;  // resting on another grain: step_sand_inner moves it

    int2 velocity = clampCellVelocity(getCellVelocity(in[i]) + (int2)(0, GRAVITY));
    int2 tmpMovement = getCellFractionalMovement(in[i]) + velocity + 128;
    int2 cellsToMove = tmpMovement / 16 - 8; // undo the +128 from above
    int2 newFrac = tmpMovement % 16;      // 128 is a multiple of 16 so the remainder is unchanged

    PathResult targetPath = claimPath(in, claimed, pos, cellsToMove, width, height);
    if (any(targetPath.end != pos + cellsToMove)) {
        int2 hitPos = targetPath.end + targetPath.blockedStep; // the cell that collided with this particle
        
        if (!isInGrid(hitPos, width, height)) {
            // Hit the edge of the grid
            if (targetPath.blockedStep.x != 0) {
                velocity.x = 0;
                newFrac.x = 0;
            }
            if (targetPath.blockedStep.y != 0) {
                velocity.y = 0;
                newFrac.y = 0;
            }
        } else {
            int hitIndex = hitPos.y * width + hitPos.x;
            uint hitCell = in[hitIndex];
            switch (getCellMaterial(hitCell)) {
                case MAT_EMPTY:
                    // Empty at the start of the step, meaning another moving grain
                    // claimed this cell first
                    hitGrain[i] = (int)claimed[hitIndex] - 1;
                    hitNormal[i] = convert_char2(targetPath.blockedStep);
                    break;

                case MAT_SAND:
                    // Another grain of sand at its start-of-step position (it may have moved away since)
                    hitGrain[i] = hitIndex;
                    hitNormal[i] = convert_char2(targetPath.blockedStep);
                    break;

                default:
                    // Stone or any other material that doesn't move
                    if (targetPath.blockedStep.x != 0) {
                        velocity.x = 0;
                        newFrac.x = 0;
                    }
                    if (targetPath.blockedStep.y != 0) {
                        velocity.y = 0;
                        newFrac.y = 0;
                    }
                    break;
            }
        }
    }

    // Update the destination and position of this grain
    int end = targetPath.end.x + targetPath.end.y * width;
    dest[i] = end;
    out[i] = MAT_EMPTY;
    out[end] = makeCell(MAT_SAND, velocity, newFrac);

    // If the grain didn't move, see whether it can move down diagonally
    if (all(targetPath.end == pos) && targetPath.blockedStep.y > 0) {
        int target = diagonalTarget(in, claimed, pos.x, pos.y, i, width, height, frame);
        if (target >= 0) {
            out[target] = makeCell(MAT_SAND, velocity, newFrac);
            out[i] = MAT_EMPTY;
            dest[i] = target;
        }
    }
}

// Simulation step for the grains of sand that have another grain directly below them.
// Runs after step_sand_bottoms, so the bottom grain of every column has already moved and
// dest[bottom] says where it went. If it fell straight down, the whole column follows it as one
// body. Otherwise the grain rests on the grain below it, or slides diagonally down.
__kernel void step_sand_inner(__global const uint* in, __global uint* out,
                              volatile __global uint* claimed,
                              __global int* hitGrain, __global char2* hitNormal,
                              __global int* dest, int width, int height, uint frame)
{
    int2 pos = (int2)((int)get_global_id(0), (int)get_global_id(1));
    if (!isInGrid(pos, width, height) || pos.y + 1 >= height)
        return;
    int i = pos.y * width + pos.x;
    int below_i = i + width;

    if (getCellMaterial(in[i]) != MAT_SAND || getCellMaterial(in[below_i]) != MAT_SAND)
        return;

    // Walk down to the bottom grain of this column of sand
    int bottom = below_i;
    while (bottom + width < width * height && getCellMaterial(in[bottom + width]) == MAT_SAND) {
        bottom += width;
    }

    int shift = dest[bottom] - bottom;  // how far the bottom grain moved, as an index offset
    if (shift > 0 && shift % width == 0) {
        // The bottom grain fell straight down by k cells: every grain of the column moves down
        // k cells too. The cells it moves into are either cells of the column itself or cells on the
        // bottom grain's path, which the bottom grain claimed, so no other grain can be there.
        int k = shift / width;
        int target = i + shift;
        out[target] = out[dest[bottom]];  // same velocity and fraction as the bottom grain
        dest[i] = target;

        // The top k grains in the column leave their cell empty. Every other grain's cell is
        // filled by the grain k cells above it, so it must not write MAT_EMPTY there.
        bool filledFromAbove = pos.y - k >= 0;
        for (int j = 1; j <= k && filledFromAbove; j++) {
            filledFromAbove = getCellMaterial(in[i - j * width]) == MAT_SAND;
        }
        if (!filledFromAbove)
            out[i] = MAT_EMPTY;
    } else {
        // The bottom grain didn't fall straight down, so this grain rests on the grain below it.
        // Resting grains have no velocity; they start falling from rest once the support is gone.
        uint resting = makeCell(MAT_SAND, (int2)(0, 0), (int2)(0, 0));
        int target = diagonalTarget(in, claimed, pos.x, pos.y, i, width, height, frame);
        if (target >= 0) {
            out[target] = resting;
            out[i] = MAT_EMPTY;
            dest[i] = target;
        } else {
            out[i] = resting;
            dest[i] = i;
        }
    }
}

// Collision pass: runs after both movement passes, so hitGrain, hitNormal and dest are complete.
// For each grain that hit another grain, adds the velocity change of the hit to both grains
// in 'impulse' (two ints per grain, x then y, indexed by start-of-step index).
__kernel void collide(__global const uint* in, __global const int* hitGrain,
                      __global const char2* hitNormal, __global const int* dest,
                      volatile __global int* impulse, int width, int height)
{
    int2 pos = (int2)((int)get_global_id(0), (int)get_global_id(1));
    if (!isInGrid(pos, width, height))
        return;
}

// Adds each grain's accumulated impulse to its velocity, in the cell it ended up in (out[dest[i]]).
__kernel void apply_impulses(__global const uint* in, __global uint* out,
                             __global const int* dest, __global const int* hitGrain,
                             __global const char2* hitNormal, __global const int* impulse,
                             int width, int height)
{
    int2 pos = (int2)((int)get_global_id(0), (int)get_global_id(1));
    if (!isInGrid(pos, width, height))
        return;
}

// Brush: puts 'material' into empty cells within 'radius' cells of (cx, cy). Each empty cell in
// the brush is filled with a chance of density / 256, so holding the brush still sprinkles grains
// instead of stamping a solid disc. Cells that aren't empty (sand, stone) are left as they are.
__kernel void paint(__global uint* cells, int width, int height, int cx, int cy, int radius,
                    uint material, uint density, uint frame)
{
    int2 pos = (int2)((int)get_global_id(0), (int)get_global_id(1));
    if (!isInGrid(pos, width, height))
        return;

    int2 d = pos - (int2)(cx, cy);
    if (d.x * d.x + d.y * d.y > radius * radius)
        return;

    int i = pos.y * width + pos.x;
    // Different seed from the step kernels, so the brush pattern doesn't line up with their random choices
    uint h = hash((uint)pos.x, (uint)pos.y, frame ^ 0x9e3779b9u);
    if (getCellMaterial(cells[i]) == MAT_EMPTY && (h & 0xFFu) < density)
        cells[i] = material;  // velocity 0, fraction 0
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

    uint material = getCellMaterial(cells[i]);
    float3 color = material < MATERIAL_COUNT ? MATERIAL_COLORS[material].xyz
                                             : (float3)(1.0f, 0.0f, 1.0f);  // magenta = unknown ID

    // Byte order R, G, B, A in memory (SDL_PIXELFORMAT_RGBA32), so R is the lowest byte.
    uint3 rgb = convert_uint3_sat_rte(color * 255.0f);
    pixels[i] = rgb.x | (rgb.y << 8) | (rgb.z << 16) | (0xFFu << 24);
}
