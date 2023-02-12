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
//#include "imgui_impl_sdl.h"
//#include "imgui_impl_sdlrenderer.h"

#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <io.h>

#include <glad\glad.h>
#include <GLFW\glfw3.h>

void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
	ImGui_ImplGlfw_CursorPosCallback(window, xposIn, yposIn);
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
}

void processInput(GLFWwindow* window);

void render(uint32_t* screen, void* cuda_pixels, const std::vector<Mesh*>& pMeshes)
{
	for (int i = 0; i < pMeshes.size(); ++i)
	{
		InitBuffers(pMeshes[i]->GetVertexPointer(), static_cast<int>(pMeshes[i]->GetVertices().size()), pMeshes[i]->GetIndices(), pMeshes[i]->GetTextures(), pMeshes[i]->GetTransform()->GetWorldTransform());
		gpuRender((uint32_t*)cuda_pixels);
	}
	if ( gpuBlit(cuda_pixels, screen) != 0 ) 
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
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow* window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Cubes", NULL, NULL);
	if (window == nullptr)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
	}

	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	uint32_t* screenData = new uint32_t[SCREEN_HEIGHT * SCREEN_WIDTH];

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

	Camera* pCamera = new Camera{ glm::vec3{ 0,0,5.f }, glm::vec3{ 0,0,-1.0f }, 45.0f, static_cast<float>(SCREEN_WIDTH) / static_cast<float>(SCREEN_HEIGHT) };

	float fpsCounter = 0.0f;

	RastizerDebugger rasterizer{ pCamera, screenData};

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

	const char* glsl_version = "#version 130";
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init(glsl_version);

	//glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetScrollCallback(window, scroll_callback);

	while (!glfwWindowShouldClose(window))
	{
		auto t2 = std::chrono::steady_clock::now();
		float elapsedSec{ std::chrono::duration<float>(t2 - t1).count() };
		fpsCounter += elapsedSec;

		t1 = t2;

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

		//SDL_LockSurface(default_screen);
		ClearDepthBuffer();

		// gpu
		if (gpu)
		{
			render(screenData, gpu_Screen, pMeshes);
			ClearScreen(gpu_Screen, glm::vec3{ 0.0f });
		}
		else
		{
			rasterizer.ClearDepthBuffer();
			rasterizer.Render(pMeshes);
			// Render gpu_Screen
		}

		// Imgui
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();

		ImGui::NewFrame();

		RenderGUI();

		ImGui::ShowDemoWindow();
		ImGui::Render();

		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glDrawPixels(SCREEN_WIDTH, SCREEN_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, screenData);
		glfwSwapBuffers(window);

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

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();

  ImGui::DestroyContext();

  for (auto mesh : pMeshes)
  {
	  delete mesh;
  }
  pMeshes.clear();

  delete pMesh;
  delete pCamera;

  	gpuFree(gpu_Screen);

    //SDL_DestroyTexture(sdlTexture);
    //SDL_DestroyRenderer(sdlRenderer);
    //SDL_DestroyWindow(window);

    //SDL_Quit();

  return 0;
}