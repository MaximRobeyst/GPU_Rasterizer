
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

#define GLM_FORCE_CUDA
#include <glm\glm.hpp>

#include "RastizerDebugger.h"

void render(SDL_Surface* screen, void* cuda_pixels, void* depth_pixels) 
{
	gpuRender((uint32_t*)cuda_pixels, (float*)depth_pixels);
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

	float* gpu_Depth = gpuAllocDepthBuffer();
	if (gpu_Depth == NULL) {
		std::cerr << "failed to alloc gpu memory" << std::endl;
	}


	std::vector<Vertex_In> triangleVertices
	{
		// Triangle 1
		Vertex_In{glm::vec3{0.0f, 1.0f, -7.5f}, glm::vec3{1.0f, 0.0f, 0.0f}},
		Vertex_In{glm::vec3{-.5f, 0.f, -7.5f}, glm::vec3{1.0f, 0.0f, 0.0f}},
		Vertex_In{glm::vec3{.5f, 0.f, -7.5f}, glm::vec3{1.0f, 0.0f, 0.0f}},

		// Triangle
		Vertex_In{glm::vec3{0.0f, 2.0f, -10.0f}, glm::vec3{1.0f, 0.0f, 0.0f}},
		Vertex_In{glm::vec3{-1.f, 0.f, -10.f}, glm::vec3{0.0f, 1.0f, 0.0f}},
		Vertex_In{glm::vec3{1.f, 0.f, -10.0f}, glm::vec3{0.0f, 0.0f, 1.0f}}
	};

	std::vector<Vertex_Out> projectedVertices{};
	projectedVertices.resize(3);

	std::vector<int> indices
	{
		0,	1,	2,
		3,	4,	5
	};

	Camera* pCamera = new Camera{ glm::vec3{ 0,0,0.f }, glm::vec3{ 0,0,-1.0f }, 45.0f, static_cast<float>(SCREEN_WIDTH) / static_cast<float>(SCREEN_HEIGHT) };

	InitBuffers(triangleVertices, indices);

	//RastizerDebugger rasterizerDebugger{ pCamera, triangleVertices, indices };
	//
	//rasterizerDebugger.Render();

    while (1) {

        SDL_Event e;
        if (SDL_PollEvent(&e)) 
		{
            if (e.type == SDL_QUIT) 
			{
                break;
            }
			if (e.type == SDL_KEYDOWN)
			{
				glm::vec3 newPosition{};
				if (e.key.keysym.sym == SDLK_w)
					newPosition.z = 1;
				if (e.key.keysym.sym == SDLK_s)
					newPosition.z = -1;
				if (e.key.keysym.sym == SDLK_d)
					newPosition.x = 1;
				if (e.key.keysym.sym == SDLK_a)
					newPosition.x = -1;
				if (e.key.keysym.sym == SDLK_LSHIFT)
					newPosition.y = 1;
				if (e.key.keysym.sym == SDLK_LCTRL)
					newPosition.y = -1;


				pCamera->UpdatePosition(newPosition);
			}

        }

		int w = SCREEN_WIDTH;
		int h = SCREEN_HEIGHT;

		uint32_t now = SDL_GetTicks();
		if (next_time_step <= now) 
		{
			gpuInit(pCamera);

			SDL_LockSurface(default_screen);
			render(default_screen, gpu_Screen, gpu_Depth);
			SDL_UnlockSurface(default_screen);

			//for (int i = 0; i < 3; ++i)
			//{
			//	glm::vec4 projectedVertex = pCamera->GetProjectionMatrix() * pCamera->GetViewMatrix() * glm::vec4(triangleVertices[i].position, 1.0f);
			//
			//	glm::vec3 normDeviceCoordinates = glm::vec3(projectedVertex.x, projectedVertex.y, projectedVertex.z) / projectedVertex.w;
			//
			//	projectedVertices[i].screenPosition =
			//		glm::vec4
			//	{
			//		((normDeviceCoordinates.x + 1) / 2) * w,
			//		((normDeviceCoordinates.y + 1) / 2) * h,
			//		normDeviceCoordinates.z,
			//		projectedVertex.w
			//	};
			//	projectedVertices[i].color = triangleVertices[i].color;
			//
			//	std::cout << "from vert ( " << triangleVertices[i].position.x << ", " << triangleVertices[i].position.y << ", " << triangleVertices[i].position.z <<
			//		") transfomed into (" << projectedVertices[i].screenPosition.x << ", " << projectedVertices[i].screenPosition.y << ", " << projectedVertices[i].screenPosition.z << ") " << std::endl;
			//}


			SDL_UpdateTexture(sdlTexture, NULL, default_screen->pixels, default_screen->pitch);
			SDL_RenderClear(sdlRenderer);
			SDL_RenderCopy(sdlRenderer, sdlTexture, NULL, NULL);
			SDL_RenderPresent(sdlRenderer);

			next_time_step += time_step;
		} else {
			SDL_Delay(next_time_step - now);
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
