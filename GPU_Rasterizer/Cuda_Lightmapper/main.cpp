#include <vld.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

//#include <imgui.h>
//#include <backends/imgui_impl_glfw.h>
//#include <backends/imgui_impl_opengl3.h>

#include <iostream>
#include <vector>

// Helper classes
#include "Material.h"
#include "Mesh.h"
#include "Texture.h"
#include "Transform.h"
#include "Camera.h"
#include "LightManager.h"

const int g_WindowWidth = 800;
const int g_WindowHeight = 600;

float g_DeltaTime = 0.0f;
float g_LastFrame = 0.0f;

inline float random(float min, float max)
{
	return min + (rand() * (int)(max - min) / RAND_MAX);
}

// Callback functions
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);


std::vector<Mesh*> pMeshes{};

void LoadScene(std::vector<Mesh*>& pMeshes, Material* pMaterial)
{
	float groundSize = 10.0f;

	// Ground plane
	Mesh* pGroundMesh = new Mesh(
	std::vector<Vertex>{
		Vertex{ glm::vec3{-groundSize, 0, groundSize}	, 	glm::vec3{0,1,0}	,glm::vec2{0,1} },
		Vertex{ glm::vec3{groundSize,0,groundSize}	,	glm::vec3{0,1,0}	,glm::vec2{1,1} },
		Vertex{ glm::vec3{groundSize,0, -groundSize}	,	glm::vec3{0,1,0}	,glm::vec2{1,0} },
		Vertex{ glm::vec3{-groundSize,0,-groundSize}	,	glm::vec3{0,1,0}	,glm::vec2{1,1} }
	},
	std::vector<unsigned int>
	{
		0,1,2,
		0,2,3
	});

	pGroundMesh->Initialize();

	Transform* pTransform = new Transform();

	pGroundMesh->SetMaterial(pMaterial);
	pGroundMesh->AddTexture(new Texture("Resources/Textures/ground.jpg"));
	pGroundMesh->SetTransform(pTransform);

	pGroundMesh->ReadyTextures();

	pMeshes.emplace_back(pGroundMesh);

	//const int nrOfCrates = 40;
	//for (int i = 0; i < nrOfCrates; ++i)
	//{
	//	// Crate Plane
	//	Mesh* pCrateMesh = Mesh::CreateCube(1.f, 1.f, 1.f);
	//
	//	float newx = random(-groundSize, groundSize);
	//	float newy = 0.5f;
	//	float newz = random(-groundSize, groundSize);
	//
	//	pCrateMesh->Initialize();
	//	Transform* pGroundTransform = new Transform(glm::vec3{ -newx, newy, newz });
	//
	//	pCrateMesh->SetMaterial(pMaterial);
	//	pCrateMesh->AddTexture(new Texture("Resources/Textures/crate.jpg"));
	//	pCrateMesh->SetTransform(pGroundTransform);
	//
	//	pCrateMesh->ReadyTextures();
	//
	//	pMeshes.emplace_back(pCrateMesh);
	//}

	LightManager::GetLightManager()->AddPointLight(PointLight{ 1.0f, 1.0f, glm::vec3{0.0f,1.0f, 0.0f}, glm::vec3{0.0f, 0.0f, 1.0f} });
}

int main()
{
	srand(time(0));

	// Init glfw
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Create window
	GLFWwindow* window = glfwCreateWindow(g_WindowWidth, g_WindowHeight, "Cuda lightmapper", NULL, NULL);
	if (window == nullptr)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	// Init callback functions
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetScrollCallback(window, scroll_callback);

	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	// load glad
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	glEnable(GL_DEPTH_TEST);


	Camera* pCamera = new Camera(90.0f, (float)g_WindowWidth / (float)g_WindowHeight, 0.1f, 100.f, new Transform(glm::vec3{ 0,1.f,3.f }));
	Material* pMaterial = new Material("Shaders/vertex.vs", "Shaders/fragment.fs");

	LoadScene(pMeshes, pMaterial);


	while (!glfwWindowShouldClose(window))
	{
		// Calculate delta time
		float currentFrame = static_cast<float>(glfwGetTime());
		g_DeltaTime = currentFrame - g_LastFrame;
		g_LastFrame = currentFrame;

		//ImGui_ImplOpenGL3_NewFrame();
		//ImGui_ImplGlfw_NewFrame();
		//ImGui::NewFrame();
		//
		//ImGui::ShowDemoWindow();

		//// 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
		//{
		//	static int counter = 0;
		//
		//	ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.
		//
		//	if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
		//		counter++;
		//	ImGui::SameLine();
		//	ImGui::Text("counter = %d", counter);
		//
		//	ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
		//	ImGui::End();
		//}
		//
		//// Rendering
		//ImGui::Render();

		// Input
		processInput(window);

		// rendering

		glBegin(GL_LINE);
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		for (int i = 0; i < pMeshes.size(); ++i)
		{
			pMeshes[i]->Draw(Camera::GetMainCamera());
		}

		glEnd();

		//ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		// buffers
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	delete Camera::GetMainCamera();
	delete pMaterial;
	delete LightManager::GetLightManager();

	for (Mesh* pMesh : pMeshes)
	{
		delete pMesh;
	}

	//ImGui_ImplOpenGL3_Shutdown();
	//ImGui_ImplGlfw_Shutdown();
	//ImGui::DestroyContext();

	glfwTerminate();
	return 0;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}

void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
	float xpos = static_cast<float>(xposIn);
	float ypos = static_cast<float>(yposIn);

	//static bool firstMouse = true;
	static float lastX = g_WindowWidth / 2, lastY = g_WindowHeight / 2;

	//if (firstMouse)
	//{
	//	lastX = xpos;
	//	lastY = ypos;
	//	firstMouse = false;
	//}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos;
	lastX = xpos;
	lastY = ypos;

	Camera::GetMainCamera()->ProcessMouseMovement(xoffset, yoffset, g_DeltaTime);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
}

void processInput(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		Camera::GetMainCamera()->ProcessKeyboard('w', g_DeltaTime);
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		Camera::GetMainCamera()->ProcessKeyboard('s', g_DeltaTime);
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		Camera::GetMainCamera()->ProcessKeyboard('a', g_DeltaTime);
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		Camera::GetMainCamera()->ProcessKeyboard('d', g_DeltaTime);

	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
		LightManager::GetLightManager()->GenerateLightmap(pMeshes);
}
