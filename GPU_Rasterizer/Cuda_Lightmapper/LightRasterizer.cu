
#include <stdint.h>
#include <builtin_types.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <device_launch_parameters.h>

#include "Mesh.h"
#include "lightMapper.cuh"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "LightRasterizer.h"

#include <algorithm>

#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <string>

#define VERTBLOCKSIZE 256
#define FRAGBLOCKSIZE 256


static glm::vec3* g_LightBuffer;
static uint32_t* g_pLightmapBuffer;

__global__
void GenerateLightmap(glm::vec3* lightBuffer, uint32_t* lightmapbuffer, int lightmapSize)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= lightmapSize) return;
}

// From Game graphics programming by Allen Sherrod
void LightRasterizer::MapVerticesToCoords(Vertex v0, Vertex v1, Vertex v2)
{
	++m_Lightmaps;
	std::vector<glm::vec2> lightMapCoords;
	lightMapCoords.resize(3);

	// Project positions
	const auto maxX = std::max(v0.position.x, std::max(v1.position.x, v2.position.x));
	const auto maxY = std::max(v0.position.y, std::max(v1.position.y, v2.position.y));
	const auto maxZ = std::max(v0.position.z, std::max(v1.position.z, v2.position.z));

	auto maxAxis = std::max(maxX, std::max(maxY, maxZ));
	if (maxAxis == maxX) m_ProjectionType = ProjectionType::XY_Projection;
	if (maxAxis == maxY) m_ProjectionType = ProjectionType::XZ_Projection;
	if (maxAxis == maxZ) m_ProjectionType = ProjectionType::YZ_Projection;

	lightMapCoords.resize(3);

	switch (m_ProjectionType)
	{
	case ProjectionType::XY_Projection:
		XYProjection(lightMapCoords[0], v0);
		XYProjection(lightMapCoords[1], v1);
		XYProjection(lightMapCoords[2], v2);
		break;
	case ProjectionType::XZ_Projection:
		XZProjection(lightMapCoords[0], v0);
		XZProjection(lightMapCoords[1], v1);
		XZProjection(lightMapCoords[2], v2);
		break;
	case ProjectionType::YZ_Projection:
		YZProjection(lightMapCoords[0], v0);
		YZProjection(lightMapCoords[1], v1);
		YZProjection(lightMapCoords[2], v2);
		break;
	}

	// Transform to texture space

	glm::vec2 minCoord = glm::min(lightMapCoords[0], glm::min(lightMapCoords[1], lightMapCoords[2]));
	glm::vec2 maxCoord = glm::max(lightMapCoords[0], glm::max(lightMapCoords[1], lightMapCoords[2]));

	glm::vec2 delta = maxCoord - minCoord;

	for (int i = 0; i < 3; ++i)
	{
		lightMapCoords[i] -= minCoord;
		lightMapCoords[i] /= delta;
	}

	// Interpolate
	auto minU = std::min(lightMapCoords[0].x, std::min(lightMapCoords[1].x, lightMapCoords[2].x));
	auto minV = std::min(lightMapCoords[0].y, std::min(lightMapCoords[1].y, lightMapCoords[2].y));

	auto maxU = std::max(lightMapCoords[0].x, std::max(lightMapCoords[1].x, lightMapCoords[2].x));
	auto maxV = std::max(lightMapCoords[0].x, std::max(lightMapCoords[1].x, lightMapCoords[2].x));

	glm::vec3 minUminV, maxUminV, minUmaxV;
	auto n = glm::normalize((v0.normal + v1.normal + v2.normal) / 3.0f);
	

	// Generate light map
	const int lightmapWidth = 16, lightmapHeight = 16;
	
	glm::vec3 lightmapImage[lightmapWidth * lightmapHeight];

	for (int y = 0; y < lightmapHeight; ++y)
	{
		for (int x = 0; x < lightmapHeight; ++x)
		{
			auto d = glm::distance(glm::vec3{(v0.position + v1.position + v2.position) / 3.0f}, v0.position);

			minUminV.x = -(n.y * minCoord.x + n.z * minCoord.y + d) / n.x;
			minUminV.y = minU;
			minUminV.z = minV;

			maxUminV.x = -(n.y * minCoord.x + n.z * minCoord.y + d) / n.x;
			maxUminV.y = maxU;
			maxUminV.z = minV;

			minUmaxV.x = -(n.y * minCoord.x + n.z * minCoord.y + d) / n.x;
			minUmaxV.y = minU;
			minUmaxV.z = maxV;

			glm::vec3 edge1 = maxUminV - minUminV;
			glm::vec3 edge2 = minUmaxV - minUminV;

			glm::vec3 edge1a = edge1 * static_cast<float>((x / lightmapWidth));
			glm::vec3 edge2a = edge2 * static_cast<float>((y / lightmapHeight));

			glm::vec3 lightmap_pos = minUminV + edge2a + edge1a;


			glm::vec3 color = glm::vec3{ 0.0f };

			LightManager* lightmanager = LightManager::GetLightManager();
			for (int lightId = 0; lightId < LightManager::GetLightManager()->GetNumberOfPointLights(); ++lightId)
			{
				glm::vec3 light_vec = lightmanager->GetPointLight(lightId).position - lightmap_pos;
				light_vec = glm::normalize(light_vec);

				float diffuse = glm::dot(n, light_vec);

				color += lightmanager->GetPointLight(lightId).color * diffuse;
			}

			lightmapImage[x + lightmapWidth * y] += color;
		}
	}

	// Turn array into image
	std::string lightmapName = "lightmap" + std::to_string(m_Lightmaps) + ".png";

	cudaFree(g_LightBuffer);
	cudaMalloc(&g_LightBuffer, lightmapHeight * lightmapWidth * sizeof(glm::vec3));
	cudaMemcpy(g_LightBuffer, &lightmapImage[0], lightmapHeight * lightmapWidth * sizeof(glm::vec3), cudaMemcpyHostToDevice);

	cudaFree(g_pLightmapBuffer);
	cudaMalloc(&g_pLightmapBuffer, lightmapHeight * lightmapWidth * sizeof(uint32_t));
	cudaMemset(g_pLightmapBuffer, 0, lightmapHeight * lightmapWidth * sizeof(uint32_t));

	const dim3 blocksPerGrid(lightmapWidth, lightmapHeight);
	const dim3 threadsPerBlock(lightmapWidth, lightmapHeight);
	GenerateLightmap<<<blocksPerGrid, threadsPerBlock>>>(g_LightBuffer, g_pLightmapBuffer, lightmapHeight);


	stbi_write_jpg(lightmapName.c_str(), lightmapWidth, lightmapHeight, 4, g_pLightmapBuffer, 100);
}

void LightRasterizer::XYProjection(glm::vec2& lighmapCoords, Vertex& vertex)
{
	lighmapCoords.x = vertex.position.x;
	lighmapCoords.y = vertex.position.y;
}

void LightRasterizer::XZProjection(glm::vec2& lighmapCoords, Vertex& vertex)
{
	lighmapCoords.x = vertex.position.x;
	lighmapCoords.y = vertex.position.z;
}

void LightRasterizer::YZProjection(glm::vec2& lighmapCoords, Vertex& vertex)
{
	lighmapCoords.x = vertex.position.y;
	lighmapCoords.y = vertex.position.z;
}
