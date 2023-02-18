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
#include "Texture.h"
#include "stb_image.h"
#include "Mesh.h"
#include "Transform.h"

#include "OBJ_Loader.h"
#include <algorithm>
#include "SDFMesh.h"

#include "imgui.h"
#include "imgui_impl_sdl.h"
#include "imgui_impl_sdlrenderer.h"

//#include "imgui_impl_glfw.h"
//#include "imgui_impl_opengl3.h"

#include <io.h>

#include <glad\glad.h>
#include <GLFW\glfw3.h>

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

void RenderGUI()
{
	static int corner = 0;
	ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav;// | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoSavedSettings;
	if (corner != -1)
	{
		constexpr float PAD = 10.0f;
		const ImGuiViewport* viewport = ImGui::GetMainViewport();
		const ImVec2 work_pos = viewport->WorkPos; // Use work area to avoid menu-bar/task-bar, if any!
		const ImVec2 work_size = viewport->WorkSize;
		ImVec2 window_pos, window_pos_pivot;
		window_pos.x = (corner & 1) ? (work_pos.x + work_size.x - PAD) : (work_pos.x + PAD);
		window_pos.y = (corner & 2) ? (work_pos.y + work_size.y - PAD) : (work_pos.y + PAD);
		window_pos_pivot.x = (corner & 1) ? 1.0f : 0.0f;
		window_pos_pivot.y = (corner & 2) ? 1.0f : 0.0f;
		ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, window_pos_pivot);
		window_flags |= ImGuiWindowFlags_NoMove;

		if (ImGui::Begin("SDF rasterizer"))
		{
			ImGui::Text("Cool imgui window");

			ImGui::End();
		}
	}
}

void LoadScene()
{

}

int main(int /*argc*/, char* /*args*/[]) 
{
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

	if (window == NULL) {
		std::cerr << "could not create window: " << SDL_GetError() << std::endl;
		return 1;
	}

	SDL_Surface* default_screen = SDL_CreateRGBSurface(0, SCREEN_WIDTH, SCREEN_HEIGHT, 32,
		0x00FF0000,
		0x0000FF00,
		0x000000FF,
		0xFF000000);

	if (default_screen == NULL) {
		SDL_Log("SDL_CreateRGBSurface() failed: %s", SDL_GetError());
		exit(1);
	}
	SDL_Renderer* sdlRenderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

	SDL_Texture* sdlTexture = SDL_CreateTexture(sdlRenderer,
		SDL_PIXELFORMAT_ARGB8888,
		SDL_TEXTUREACCESS_STREAMING | SDL_TEXTUREACCESS_TARGET,
		SCREEN_WIDTH, SCREEN_HEIGHT);


	if (sdlTexture == NULL) {
		SDL_Log("SDL_Error failed: %s", SDL_GetError());
		exit(1);
	}

	uint32_t* gpu_Screen = gpuAllocScreenBuffer();	
	if ( gpu_Screen == NULL ) {
		std::cerr << "failed to alloc gpu memory" << std::endl;
	}

	objl::Loader loader;

	std::vector<Mesh*> pMeshes{};

	std::vector<Vertex_In> triangleVertices{};
	std::vector<unsigned int> indices{};

	bool success = loader.LoadFile("Resources/tuktuk.obj");
	if (!success)
	{
		std::cout << "File not found" << std::endl;
		exit(-1);
	}

	auto storeVertex = [&](objl::Vertex vertex) {
		triangleVertices.emplace_back(
			Vertex_In{
				glm::vec3{ vertex.Position.X, vertex.Position.Y, vertex.Position.Z },
				glm::vec3{vertex.Normal.X, vertex.Normal.Y, vertex.Normal.Z},
				glm::vec3{1,1,1},
				glm::vec2{vertex.TextureCoordinate.X, vertex.TextureCoordinate.Y}
			});
	};

	for (int i = 0; i < loader.LoadedMeshes.size(); ++i)
	{
		triangleVertices.clear();
		triangleVertices.reserve(loader.LoadedMeshes[i].Vertices.size());
		std::for_each(loader.LoadedMeshes[i].Vertices.begin(), loader.LoadedMeshes[i].Vertices.end(), storeVertex);
		pMeshes.emplace_back(new Mesh(triangleVertices, loader.LoadedMeshes[i].Indices));

		//pMeshes[pMeshes.size() - 1]->AddTexture(new Texture{ "Resources/tuktuk.jpg" });
		pMeshes[pMeshes.size() - 1]->SetTransform(new Transform{ glm::vec3{0.0f,0.0f,0} });
	}

	SDFMesh* pMesh = new SDFMesh(pMeshes[pMeshes.size() - 1], 16);

	Camera* pCamera = new Camera{ glm::vec3{ 0,0,75.f }, glm::vec3{ 0,0,-1.0f }, 45.0f, static_cast<float>(SCREEN_WIDTH) / static_cast<float>(SCREEN_HEIGHT) };

	float fpsCounter = 0.0f;

	RastizerDebugger rasterizer{ pCamera, (uint32_t*)default_screen->pixels};

	auto t1 = std::chrono::steady_clock::now();

	float t = 0.0f;
	bool paused = true;
	bool gpu = false;

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
	io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

	ImGui::StyleColorsDark();

	ImGuiStyle& style = ImGui::GetStyle();
	if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
	{
		style.WindowRounding = 0.0f;
		style.Colors[ImGuiCol_WindowBg].w = 1.0f;
	}

	ImGui_ImplSDL2_InitForSDLRenderer(window, sdlRenderer);
	ImGui_ImplSDLRenderer_Init(sdlRenderer);

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
				t += 0.25f * elapsedSec;
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


		// Imgui
		ImGui_ImplSDLRenderer_NewFrame();
		ImGui_ImplSDL2_NewFrame(window);

		ImGui::NewFrame();

		RenderGUI();

		ImGui::ShowDemoWindow();
		ImGui::Render();

		ImGui_ImplSDLRenderer_RenderDrawData(ImGui::GetDrawData());

		// Switch buffers
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

  ImGui_ImplSDLRenderer_Shutdown();
  ImGui_ImplSDL2_Shutdown();

  ImGui::DestroyContext();

  for (auto mesh : pMeshes)
  {
	  delete mesh;
  }
  pMeshes.clear();

  delete pMesh;
  delete pCamera;

  	gpuFree(gpu_Screen);

    SDL_DestroyTexture(sdlTexture);
    SDL_DestroyRenderer(sdlRenderer);
    SDL_DestroyWindow(window);

    SDL_Quit();

  return 0;
}