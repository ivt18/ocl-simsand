# ocl-simsand

A foundation for a GPU falling-sand simulation. OpenCL runs the simulation on any GPU vendor
(NVIDIA, AMD, Intel), and SDL3 opens the window and displays the result.

The program starts with a placeholder scene: a stone floor and a block of sand. Hold the left mouse
button to add sand at the cursor, and use the mouse wheel to change the brush size.

## Dependencies

Install these yourself:

| Dependency | Notes |
|---|---|
| A GPU driver with OpenCL support | Standard NVIDIA, AMD and Intel drivers include it. No separate SDK is needed. |
| [CMake](https://cmake.org/download/) 3.24 or newer | |
| Visual Studio or Build Tools 2022/2026 | Install the **Desktop development with C++** workload. |
| [Git for Windows](https://git-scm.com/download/win) | CMake uses it to download dependencies, and it provides Git Bash for `windows_build.sh`. |

CMake downloads these automatically the first time you configure:

- [SDL3](https://github.com/libsdl-org/SDL) 3.4.18: window, input and drawing (built as a static library)
- [OpenCL-Headers](https://github.com/KhronosGroup/OpenCL-Headers), [OpenCL-ICD-Loader](https://github.com/KhronosGroup/OpenCL-ICD-Loader)
  and [OpenCL-CLHPP](https://github.com/KhronosGroup/OpenCL-CLHPP) (C++ bindings), all v2026.05.29

## Building and running

From cmd or PowerShell (the configuration defaults to Debug):

```bat
.\windows_build.bat
.\windows_build.bat Release
.\build\Debug\ocl-simsand.exe
```

Or from Git Bash:

```sh
./windows_build.sh            # Debug
./windows_build.sh Release
./build/Debug/ocl-simsand.exe
```

Or with CMake directly:

```sh
cmake -S . -B build -A x64
cmake --build build --config Debug
```

On startup the program prints the OpenCL device it picked. Esc closes the window.

**Kernels are compiled when the program starts**, from `kernels/simulate.cl` in the source folder.
Editing a kernel only needs a restart, not a rebuild. If a kernel fails to compile, the program
prints the compiler's error log and exits.

## Project structure

```
CMakeLists.txt        Build: dependencies and sources
windows_build.bat     Build script for cmd/PowerShell on Windows
windows_build.sh      Build script for Git Bash on Windows
kernels/
  simulate.cl         OpenCL kernels: simulation, colorize (cells → pixels), and painting with the brush
  shared.h            Constants shared by C++ and the kernels: material IDs, work-group size
src/
  main.cpp            Window, main loop, test scene, grid and window sizes
  simulation.h/.cpp   Cell buffers (ping-pong) and kernel launches
  cl_setup.h/.cpp     Picks an OpenCL device, creates the context and queue, compiles kernels
```

## How a frame works

```
main loop
  sim.paint()           while the left mouse button is held: paint kernel adds sand around the cursor
  sim.step()            one simulation step (below), then cells_[current] and cells_[other] swap
  sim.readPixels()      colorize kernel: cell → RGBA pixel; copy pixels to the CPU
  SDL                   upload pixels to a texture and draw it, scaled to the window

sim.step()
  copy                  cells_[other] = cells_[current], so cells nobody moves keep their contents
  clear                 claimed = 0, hitGrain = -1, impulse = 0
  step_sand_bottoms     grains without sand directly below: gravity, then walk the path with claimPath
  step_sand_inner       grains resting on sand: follow their column's bottom grain, rest, or slide diagonally
  collide               collision response between grains (stub, not implemented yet)
  apply_impulses        adds the collision response to the grains' velocities (stub, not implemented yet)
```

- **Cells:** one `uint` per cell, row-major (`y * width + x`). The bit layout is defined in `shared.h`:

  | Bits | Field | Range |
  |---|---|---|
  | 0–7 | material ID (`MAT_*`) | 0..255 |
  | 8–15 | x velocity | -128..127, in 16ths of a cell per step |
  | 16–23 | y velocity | -128..127, in 16ths of a cell per step |
  | 24–27 | x position within the cell | 0..15, in 16ths of a cell |
  | 28–31 | y position within the cell | 0..15, in 16ths of a cell |

  Use the helpers in `simulate.cl` (`getCellMaterial`, `getCellVelocity`, `getCellFractionalMovement`,
  `makeCell`) rather than comparing whole cells: a moving grain of sand is not equal to `MAT_SAND`.
- **Ping-pong:** each step reads one buffer (`in`) and writes the other (`out`), so every work-item
  reads the same snapshot of the grid. `out` starts as a copy of `in`, so a kernel only writes the
  cells that change.
- **Claims:** a grain may only move into a cell it has claimed with `atomic_cmpxchg` in the
  `claimed` buffer, so two grains never end up in the same cell. Each grain is identified by its
  index at the start of the step; `dest` records where it ended up.
- **Columns:** a grain can only move into cells that were empty at the start of the step. So
  `step_sand_bottoms` moves the bottom grain of each column first, and `step_sand_inner` then moves
  the grains above it by the same distance. That keeps falling sand packed together.
- **Ordering:** everything goes through one in-order command queue, like a single CUDA stream.
  Each kernel starts only after the previous one has finished, which is what lets a pass read what
  the pass before it wrote. The blocking read in `readPixels` waits for all of them.
- **Scaling and input:** `SDL_SetRenderLogicalPresentation` draws the grid at its own resolution and
  scales it to fit the window with black bars. `SDL_ConvertEventToRenderCoordinates` converts mouse
  positions to cell coordinates for the brush. The mouse wheel changes the brush size.

## Coming from CUDA

| CUDA | OpenCL |
|---|---|
| `__global__ void k(...)` | `__kernel void k(...)` |
| `blockIdx * blockDim + threadIdx` | `get_global_id(dim)` |
| thread block / `blockDim` | work-group / local size (`WORKGROUP_SIZE`) |
| `cudaMalloc` | `cl::Buffer` |
| `k<<<grid, block>>>(args)` | `kernel(cl::EnqueueArgs(queue, global, local), args)` |
| stream | command queue |
| `cudaMemcpy` | `enqueueReadBuffer` / `enqueueWriteBuffer` |
| `__shared__`, `__syncthreads()` | `__local`, `barrier(CLK_LOCAL_MEM_FENCE)` |

Note that OpenCL's *global* size is the **total** number of work-items, not the number of groups.