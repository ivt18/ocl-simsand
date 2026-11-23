#include "shared.h"
#include "simulation.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef SIMSAND_KERNEL_DIR
#define SIMSAND_KERNEL_DIR "kernels"
#endif

namespace {

constexpr int kGridWidth = 640;
constexpr int kGridHeight = 360;
constexpr int kWindowWidth = 1280;
constexpr int kWindowHeight = 720;

// PLACEHOLDER: a starting scene so there is something on screen: a stone floor and a block of sand.
std::vector<cl_uint> makeTestScene(int width, int height) {
    std::vector<cl_uint> cells(size_t(width) * height, MAT_EMPTY);
    for (int y = height - 8; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            cells[y * width + x] = MAT_STONE;
        }
    }
    for (int y = height / 4; y < height / 4 + 40; ++y) {
        for (int x = width / 2 - 40; x < width / 2 + 40; ++x) {
            cells[y * width + x] = MAT_SAND;
        }
    }
    return cells;
}

void sdlCheck(bool ok, const char* what) {
    if (!ok) {
        throw std::runtime_error(std::string(what) + " failed: " + SDL_GetError());
    }
}

void run() {
    sdlCheck(SDL_Init(SDL_INIT_VIDEO), "SDL_Init");

    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    sdlCheck(SDL_CreateWindowAndRenderer("ocl-simsand", kWindowWidth, kWindowHeight, SDL_WINDOW_RESIZABLE,
                                         &window, &renderer),
             "SDL_CreateWindowAndRenderer");
    SDL_SetRenderVSync(renderer, 1);

    // Draw in grid coordinates (one unit = one cell). SDL scales that to the window,
    // keeping the aspect ratio and filling the rest with black bars.
    SDL_SetRenderLogicalPresentation(renderer, kGridWidth, kGridHeight, SDL_LOGICAL_PRESENTATION_LETTERBOX);

    // One texel per cell; nearest-neighbour scaling keeps cells crisp instead of blurred.
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING,
                                             kGridWidth, kGridHeight);
    sdlCheck(texture != nullptr, "SDL_CreateTexture");
    SDL_SetTextureScaleMode(texture, SDL_SCALEMODE_NEAREST);

    const ClSetup cl = createClSetup();
    Simulation sim(cl, kGridWidth, kGridHeight, SIMSAND_KERNEL_DIR);
    sim.upload(makeTestScene(kGridWidth, kGridHeight));

    std::vector<uint32_t> pixels;
    uint64_t titleTime = SDL_GetTicks();
    int framesSinceTitle = 0;

    bool running = true;
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            // Tip: SDL_ConvertEventToRenderCoordinates(renderer, &event) turns mouse positions
            // into cell coordinates, because of the logical presentation set above.
            if (event.type == SDL_EVENT_QUIT) {
                running = false;
            } else if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_ESCAPE) {
                running = false;
            }
        }

        // 1. Simulate on the GPU, then fetch the image.
        sim.step();
        sim.readPixels(pixels);

        // 2. Show it.
        SDL_UpdateTexture(texture, nullptr, pixels.data(), kGridWidth * int(sizeof(uint32_t)));
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);
        SDL_RenderTexture(renderer, texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);  // waits for vsync

        ++framesSinceTitle;
        const uint64_t now = SDL_GetTicks();
        if (now - titleTime >= 500) {
            char title[64];
            std::snprintf(title, sizeof(title), "ocl-simsand - %.0f fps",
                          framesSinceTitle * 1000.0 / double(now - titleTime));
            SDL_SetWindowTitle(window, title);
            titleTime = now;
            framesSinceTitle = 0;
        }
    }

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

} // namespace

int main(int /*argc*/, char* /*argv*/[]) {
    try {
        run();
    } catch (const cl::Error& e) {
        std::fprintf(stderr, "OpenCL error %d in %s\n", e.err(), e.what());
        return 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Fatal error: %s\n", e.what());
        return 1;
    }
    return 0;
}
