
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

#define VERTBLOCKSIZE 256
#define FRAGBLOCKSIZE 256

__global__
void VertexShader(int size, Vertex* vertexInBuffer, Lightmapper_Vert* vertexOutBuffer, glm::mat4 world, glm::mat4 projection)
{
	int index = ((blockIdx.x * blockDim.x) + threadIdx.x) + (((blockIdx.y * blockDim.y) + threadIdx.y) * size);

	if (index >= g_VertCount) return;

	glm::vec4 projectedVertex = projection * world * glm::vec4(vertexInBuffer[index].position, 1.0f);

	glm::vec3 normDeviceCoordinates = glm::vec3(projectedVertex.x, projectedVertex.y, projectedVertex.z) / projectedVertex.w;

	vertexOutBuffer[index].lightmapperPosition =
		glm::vec4
	{
		((normDeviceCoordinates.x + 1) / 2) * size,
		((1 - normDeviceCoordinates.y) / 2) * size,
		normDeviceCoordinates.z,
		projectedVertex.w
	};
	vertexOutBuffer[index].uv = vertexInBuffer[index].uv;
}

void DirectionalLightRender(DirectionalLight light, uint32_t* buffer)
{
	int vertexBlockSize = VERTBLOCKSIZE, fragmentBlockSize = FRAGBLOCKSIZE;
	int vertexGridSize = (g_VertCount + VERTBLOCKSIZE - 1) / VERTBLOCKSIZE;

	glm::mat4 matrix = glm::ortho(0.0f, 1000.f, 0.0f, 1000.f);


	glm::vec3 right = glm::normalize(glm::cross(light.direction, glm::vec3{ 0,1,0 }));
	glm::vec3 up = glm::normalize(glm::cross(right, light.direction));

	glm::mat4 worldMatrix = glm::lookAt(-light.direction * 100.0f, (-light.direction * 100.0f) + light.direction, up);

	VertexShader << <vertexGridSize, vertexBlockSize >> > (g_LightmapSize, g_pVerteInBuffer, g_pVerteOutBuffer, worldMatrix, matrix);
}