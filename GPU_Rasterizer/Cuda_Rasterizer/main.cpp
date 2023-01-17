
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
#pragma warning(push, 0)
#include <glm\glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#pragma warning(pop)

#include "RastizerDebugger.h"
#include "EOBJParser.h"
#include "Texture.h"
#include "stb_image.h"
#include "Mesh.h"
#include "Transform.h"

void render(SDL_Surface* screen, void* cuda_pixels, const std::vector<Mesh*>& pMeshes) 
{
	for (int i = 0; i < pMeshes.size(); ++i)
	{
		InitBuffers(pMeshes[i]->GetVertexPointer(), static_cast<int>(pMeshes[i]->GetVertices().size()), pMeshes[i]->GetIndices(), pMeshes[i]->GetTextures(), pMeshes[i]->GetTransform()->GetWorldTransform());
		gpuRender((uint32_t*)cuda_pixels);
	}
	if ( gpuBlit(cuda_pixels, screen->pixels) != 0 ) 
	{
		cudaError_t err{ cudaGetLastError() };

		// todo: get cuda error
		std::cerr << cudaGetErrorString(err) << std::endl;
	};
}

int main(int /*argc*/, char* /*args*/[]) {

	uint32_t time_step{ static_cast<uint32_t>(1000. / 60.) };
	uint32_t next_time_step{ SDL_GetTicks() };

	//stbi_set_flip_vertically_on_load(true);

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

	std::vector<Mesh*> pMeshes{};

	std::vector<Vertex_In> triangleVertices{};
		//{
		//	// Triangle 1
		//	Vertex_In{glm::vec3{.5f, 0.5f, 0.f}, glm::vec3{0,0,1}, glm::vec3{1,0,0}, glm::vec3{1.0f, 0.0f, 0.0f}, glm::vec2{1,1}},
		//	Vertex_In{glm::vec3{-.5f, -.5f, 0.f}, glm::vec3{0,0,1}, glm::vec3{1,0,0}, glm::vec3{0.0f, 1.0f, 0.0f}, glm::vec2{0,0}},
		//	Vertex_In{glm::vec3{.5f, -.5f, 0.f}, glm::vec3{0,0,1}, glm::vec3{1,0,0}, glm::vec3{0.0f, 0.0f, 1.0f}, glm::vec2{1,0}},
		//
		//	// Triangle 2
		//	Vertex_In{glm::vec3{.5f, 0.5f, 0.f}, glm::vec3{0,0,1}, glm::vec3{1,0,0}, glm::vec3{1.0f, 0.0f, 0.0f}, glm::vec2{1,1}},
		//	Vertex_In{glm::vec3{-.5f, -.5f, 0.f}, glm::vec3{0,0,1}, glm::vec3{1,0,0}, glm::vec3{0.0f, 1.0f, 0.0f}, glm::vec2{0,0}},
		//	Vertex_In{glm::vec3{-.5f, 0.5f, 0.f}, glm::vec3{0,0,1}, glm::vec3{1,0,0}, glm::vec3{0.0f, 0.0f, 0.0f}, glm::vec2{0,1}}
		//};
		//
		std::vector<unsigned int> indices{};
	//{
	//	0,	1,	2,
	//	5, 4, 3,
	//};
	//
	//pMeshes.emplace_back(new Mesh(triangleVertices, indices));
	//pMeshes[pMeshes.size() - 1]->SetTransform(new Transform{ glm::vec3{0.0f,0,0} });

	Elite::ParseOBJ("Resources/tuktuk.obj", triangleVertices, indices);
	pMeshes.emplace_back(new Mesh( triangleVertices, indices ));
	pMeshes[pMeshes.size() - 1]->AddTexture(new Texture{ "Resources/tuktuk.png" });
	pMeshes[pMeshes.size() - 1]->SetTransform(new Transform{ glm::vec3{-10.0f,0.0f,0} });
	
	Elite::ParseOBJ("Resources/vehicle.obj", triangleVertices, indices);
	pMeshes.emplace_back(new Mesh(triangleVertices, indices));
	pMeshes[pMeshes.size() - 1]->AddTexture(new Texture{ "Resources/vehicle_diffuse.png" });
	pMeshes[pMeshes.size() - 1]->SetTransform(new Transform{ glm::vec3{10.0f,0,0} ,  glm::vec3{0.5f} });

	Camera* pCamera = new Camera{ glm::vec3{ 0,0,5.f }, glm::vec3{ 0,0,-1.0f }, 45.0f, static_cast<float>(SCREEN_WIDTH) / static_cast<float>(SCREEN_HEIGHT) };

	//InitBuffers(triangleVertices, indices, textures);
	float fpsCounter = 0.0f;

	RastizerDebugger rasterizer{ pCamera, (uint32_t*)default_screen ->pixels};

	auto t1 = std::chrono::steady_clock::now();

	float t = 0.0f;
	bool paused = true;
	bool gpu = false;

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
				cudaDeviceSynchronize();
		        break;
		    }
			if (e.type == SDL_KEYDOWN)
			{
				if (e.key.keysym.sym == SDLK_p)
					paused = !paused;
				if (e.key.keysym.sym == SDLK_r)
				{
					gpu = !gpu;
					if (gpu)
						std::cout << "GPU rendering" << std::endl;
					else
						std::cout << "CPU Rendering" << std::endl;
				}
			}
		}

		pCamera->Update(elapsedSec);
		for (auto mesh : pMeshes)
		{
			if (!paused)
			{
				mesh->GetTransform()->SetRotation(0, t, 0);
				t += 1.0f * elapsedSec;
			}
		}

		gpuInit(pCamera);

		SDL_LockSurface(default_screen);
		ClearDepthBuffer();

		// gpu
		if (gpu)
		{
			render(default_screen, gpu_Screen, pMeshes);
			ClearScreen(gpu_Screen, glm::vec3{ 0.0f });
		}
		else
		{
			rasterizer.ClearDepthBuffer();
			rasterizer.Render(pMeshes);
			// Render gpu_Screen
		}
		//rasterizer.Render();
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

  for (auto mesh : pMeshes)
  {
	  delete mesh;
  }
  pMeshes.clear();

  delete pCamera;

  	gpuFree(gpu_Screen);

    SDL_DestroyTexture(sdlTexture);
    SDL_DestroyRenderer(sdlRenderer);
    SDL_DestroyWindow(window);

    SDL_Quit();

  return 0;
}
