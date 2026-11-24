# ocl-simsand

A foundation for a GPU falling-sand simulation. OpenCL runs the simulation on any GPU vendor
(NVIDIA, AMD, Intel), and SDL3 opens the window and displays the result.

Right now the simulation step only copies the grid unchanged, and the program shows a test scene:
a stone floor and a block of sand. The sand rules are still to be written.

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
  simulate.cl         OpenCL kernels: step (the simulation) and colorize (cells → pixels)
  shared.h            Constants shared by C++ and the kernels: material IDs, work-group size
src/
  main.cpp            Window, main loop, test scene, grid and window sizes
  simulation.h/.cpp   Cell buffers (ping-pong) and kernel launches
  cl_setup.h/.cpp     Picks an OpenCL device, creates the context and queue, compiles kernels
```

## How a frame works

```
main loop
  sim.step()          step kernel: reads cells_[current], writes cells_[other], then they swap
  sim.readPixels()    colorize kernel: material ID → RGBA pixel; copy pixels to the CPU
  SDL                 upload pixels to a texture and draw it, scaled to the window
```

- **Cells:** one `uint` per cell, row-major (`y * width + x`), holding a material ID from `shared.h`.
- **Ping-pong:** each step reads one buffer and writes the other, so it always reads a stable
  snapshot. The kernel must write every in-bounds cell on every step.
- **Ordering:** everything goes through one in-order command queue, like a single CUDA stream.
  Kernels run one after another, and the blocking read in `readPixels` waits for them.
- **Scaling:** `SDL_SetRenderLogicalPresentation` draws the grid at its own resolution and scales
  it to fit the window with black bars. `SDL_ConvertEventToRenderCoordinates` converts mouse
  positions to cell coordinates.

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

## Where to start

- **Simulation rules:** the `step` kernel in `kernels/simulate.cl`.
- **Materials and colours:** IDs in `kernels/shared.h`, colours in `MATERIAL_COLORS` in `simulate.cl`.
  Use `float4` (not `float3`) for arrays and buffers: `float3` layouts differ between compilers.
- **Starting scene, grid size, window size:** `src/main.cpp`.
- **Input:** the event loop in `src/main.cpp`.
