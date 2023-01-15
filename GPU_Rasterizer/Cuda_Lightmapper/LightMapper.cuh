#include "LightManager.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <device_launch_parameters.h>

#include "checkCUDAError.h"

#include "Mesh.h"

struct Lightmapper_Vert
{
	glm::vec4 lightmapperPosition;
	glm::vec4 lightmapperColor;
	glm::vec3 position;
	glm::vec2 uv;
};


// Lights
static PointLight* g_pPointLights = NULL;
static DirectionalLight* g_pDirectionalLights = NULL;
static int m_PointlightsCount;
static int m_DirectionalLightsCount;

// Mesh info
static Vertex* g_pVertices;
static unsigned int* g_pIndices;

// Textures
static uint32_t* g_pLightmap;
static int g_LightmapSize;

// Vertex buffer
static Vertex* g_pVerteInBuffer = NULL;
static Lightmapper_Vert* g_pVerteOutBuffer = NULL;
static unsigned int* g_pIndexBuffer;

static int g_VertCount;
static int g_IndicesCount;

__inline__
void InitPointLights(const std::vector<PointLight>& pointLights)
{
	m_PointlightsCount = pointLights.size();

	cudaFree(g_pPointLights);
	cudaMalloc(&g_pPointLights, pointLights.size() * sizeof(PointLight));
	cudaMemcpy(g_pPointLights, pointLights.data(), pointLights.size() * sizeof(PointLight), cudaMemcpyHostToDevice);

	checkCUDAError("InitPointLights");
}

__inline__
void InitDirectionalLights(const std::vector<DirectionalLight>& directionLights)
{
	m_DirectionalLightsCount = directionLights.size();

	cudaFree(g_pDirectionalLights);
	cudaMalloc(&g_pDirectionalLights, directionLights.size() * sizeof(DirectionalLight));
	cudaMemcpy(g_pDirectionalLights, directionLights.data(), directionLights.size() * sizeof(DirectionalLight), cudaMemcpyHostToDevice);

	checkCUDAError("InitDirectionalLights");
}

__inline__
void InitLightMap(int size = 128)
{
	g_LightmapSize = size;

	cudaFree(g_pLightmap);
	cudaMalloc(&g_pLightmap, size * size * sizeof(uint32_t));
	cudaMemset(g_pLightmap, 0, size * size * sizeof(uint32_t));
}
