
#define SDL_MAIN_HANDLED
#include <SDL.h>
#include <stdio.h>
#include <iostream>

#include <string.h>

#include "const.h"
#include "gpu.h"

#include <vld.h>

#include "Camera.h"
#include <cuda_runtime_api.h>
#include <glm/gtc/type_ptr.hpp>

#include <vector>
#include <chrono>

#define GLM_FORCE_CUDA
#include <glm\glm.hpp>

#include "RastizerDebugger.h"
#include "EOBJParser.h"
#include "Texture.h"

void render(SDL_Surface* screen, void* cuda_pixels) 
{
	gpuRender((uint32_t*)cuda_pixels);
	if ( gpuBlit(cuda_pixels, screen->pixels) != 0 ) 
	{
		cudaError_t err{ cudaGetLastError() };

		// todo: get cuda error
		std::cerr << cudaGetErrorString(err) << std::endl;
	};
}

int main(int argc, char* args[]) {

	uint32_t time_step = 1000. / 60. ;
	uint32_t next_time_step = SDL_GetTicks();

	if (SDL_Init(SDL_INIT_EVERYTHING) < 0) {
		std::cerr << "could not initialize sdl2: " << SDL_GetError() << std::endl;
		return 1;
	}

	SDL_Window* window = SDL_CreateWindow(
				"main",
				SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
				SCREEN_WIDTH, SCREEN_HEIGHT,
				SDL_WINDOW_SHOWN
				);

	SDL_Surface* default_screen = SDL_CreateRGBSurface( 0, SCREEN_WIDTH, SCREEN_HEIGHT, 32,
												0x00FF0000,
												0x0000FF00,
												0x000000FF,
												0xFF000000);

	if (default_screen == NULL) {
        SDL_Log("SDL_CreateRGBSurface() failed: %s", SDL_GetError());
        exit(1);
	}

    SDL_Renderer* sdlRenderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

	SDL_Texture *sdlTexture = SDL_CreateTexture(sdlRenderer,
												SDL_PIXELFORMAT_ARGB8888,
												SDL_TEXTUREACCESS_STREAMING | SDL_TEXTUREACCESS_TARGET,
												SCREEN_WIDTH, SCREEN_HEIGHT);

	if (sdlTexture== NULL) {
        SDL_Log("SDL_Error failed: %s", SDL_GetError());
        exit(1);
	}

	uint32_t* gpu_Screen = gpuAllocScreenBuffer();	
	if ( gpu_Screen == NULL ) {
		std::cerr << "failed to alloc gpu memory" << std::endl;
	}

	//float* gpu_Depth = gpuAllocDepthBuffer();
	//if (gpu_Depth == NULL) {
	//	std::cerr << "failed to alloc gpu memory" << std::endl;
	//}


	std::vector<Vertex_In> triangleVertices
	{
		// Triangle 1
		Vertex_In{glm::vec3{0.0f, 1.0f, -7.5f}, glm::vec3{1.0f, 1.0f, 1.0f}},
		Vertex_In{glm::vec3{-.5f, 0.f, -7.5f}, glm::vec3{1.0f, 1.0f, 1.0f}},
		Vertex_In{glm::vec3{.5f, 0.f, -7.5f}, glm::vec3{1.0f, 1.0f, 1.0f}},
	
		// Triangle
		Vertex_In{glm::vec3{0.0f, 2.0f, -10.0f}, glm::vec3{1.0f, 0.0f, 0.0f}},
		Vertex_In{glm::vec3{-1.f, 0.f, -10.f}, glm::vec3{0.0f, 1.0f, 0.0f}},
		Vertex_In{glm::vec3{1.f, 0.f, -10.0f}, glm::vec3{0.0f, 0.0f, 1.0f}}
	};

	std::vector<int> indices
	{
		0,	1,	2,
		3,	4,	5
	};

	Elite::ParseOBJ("Resources/tuktuk.obj", triangleVertices, indices);

	std::vector<Texture> textures;
	textures.emplace_back(Texture{ "Resources/tuktuk.png" });

	Camera* pCamera = new Camera{ glm::vec3{ 0,5,64.f }, glm::vec3{ 0,0,-1.0f }, 45.0f, static_cast<float>(SCREEN_WIDTH) / static_cast<float>(SCREEN_HEIGHT) };

	InitBuffers(triangleVertices, indices, textures);
	float elapsedSec = 0.0f;
	float fpsCounter = 0.0f;

	//RastizerDebugger rasterizer{ pCamera, triangleVertices, indices };
	//rasterizer.Render();

	auto t1 = std::chrono::steady_clock::now();

	while (1)
	{
		auto t2 = std::chrono::steady_clock::now();
		float elapsedSec{ std::chrono::duration<float>(t2 - t1).count() };
		fpsCounter += elapsedSec;

		t1 = t2;

		SDL_Event e;
		if (SDL_PollEvent(&e)) 
		{
		    if (e.type == SDL_QUIT) 
			{
		        break;
		    }
		}

		pCamera->Update(elapsedSec);
		gpuInit(pCamera);

		SDL_LockSurface(default_screen);
		render(default_screen, gpu_Screen);
		SDL_UnlockSurface(default_screen);

		SDL_UpdateTexture(sdlTexture, NULL, default_screen->pixels, default_screen->pitch);
		SDL_RenderClear(sdlRenderer);
		SDL_RenderCopy(sdlRenderer, sdlTexture, NULL, NULL);
		SDL_RenderPresent(sdlRenderer);

		if (fpsCounter >= 1.0f)
		{
			std::cout << "FPS: " << 1.0f / elapsedSec << std::endl;
			fpsCounter -= 1.0f;
		}
	}

  if (window == NULL) {
	  std::cerr << "could not create window: " << SDL_GetError() << std::endl;
    return 1;
  }
	
  delete pCamera;

  	gpuFree(gpu_Screen);

    SDL_DestroyTexture(sdlTexture);
    SDL_DestroyRenderer(sdlRenderer);
    SDL_DestroyWindow(window);

    SDL_Quit();

  return 0;
}
