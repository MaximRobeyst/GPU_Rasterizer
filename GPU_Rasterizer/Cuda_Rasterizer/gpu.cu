#include "gpu.h"
#include <driver_types.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "Camera.h"


__device__ float g_WorldMatri[4][4];
__device__ float g_ProjectionMatri[4][4];

void gpuInit(Camera* pCamera)
{
	cudaMemcpy(g_WorldMatri, glm::value_ptr(pCamera->GetWorldMatrix()), sizeof(glm::mat4), cudaMemcpyDefault);
	cudaMemcpy(g_ProjectionMatri, glm::value_ptr(pCamera->GetProjectionMatrix()), sizeof(glm::mat4), cudaMemcpyDefault);
}


uint32_t * gpuAlloc(void) {
	uint32_t* gpu_mem;

	cudaError_t err = cudaMalloc(&gpu_mem, SCREEN_SIZE * 4);
	if ( err != cudaSuccess ) return NULL;

	return gpu_mem;
};

void gpuFree(void* gpu_mem) {
	cudaFree(gpu_mem);
}

int gpuBlit(void* src, void* dst){
	cudaError_t err = cudaMemcpy(dst, src, SCREEN_SIZE * 4, cudaMemcpyDeviceToHost);
	if ( err != cudaSuccess ) 
		return 1;
	return 0;
}

// ----- 
__host__
__device__
inline float Cross(const glm::vec2& v1, const glm::vec2& v2)
{
	return v1.x * v2.y - v1.y * v2.x;
}

__host__
__device__
void TransformVertex(const glm::vec3& originalVertex, glm::vec4& transformedVertex, const glm::mat4& worldToView, const glm::mat4& projectionMatrix)
{
	glm::vec4 projectedVertex = worldToView * glm::vec4(originalVertex, 1.0f);
	projectedVertex = projectionMatrix * projectedVertex;

	projectedVertex.x /= projectedVertex.w;
	projectedVertex.y /= projectedVertex.w;
	projectedVertex.z /= projectedVertex.w;

	projectedVertex.x = ((projectedVertex.x + 1) / 2) * static_cast<float>(SCREEN_WIDTH);
	projectedVertex.y = ((1 - projectedVertex.y) / 2) * static_cast<float>(SCREEN_HEIGHT);

	float vector[4]{ projectedVertex.x, projectedVertex.y, projectedVertex.z, projectedVertex.w };

	transformedVertex = projectedVertex;
}

__host__
__device__
void TransformVertices(const glm::vec3* originalVertex, glm::vec4* transformedVertex, int size, const glm::mat4& worldToView, const glm::mat4& projectionMatrix)
{
	for (size_t vertexIndex = 0; vertexIndex < size; ++vertexIndex)
	{
		TransformVertex(originalVertex[vertexIndex], transformedVertex[vertexIndex], worldToView, projectionMatrix);
	}
}

__host__
__device__
uint32_t getPixColor(int x, int y) {

	glm::vec2 pixel = glm::vec2{ static_cast<float>(x), static_cast<float>(y) };
	glm::vec4 screenSpaceVertices[3];

	glm::vec3 triangleVertices[3]
	{
		glm::vec3{0.0f, 2.0f, 0.0f},
		glm::vec3{-1.f, 0.f, 0.f},
		glm::vec3{1.f, 0.f, 0.0f}
	};


	glm::mat4 worldMatri = glm::make_mat4(&g_WorldMatri[0][0]);
	glm::mat4 projectionMatri = glm::make_mat4(&g_ProjectionMatri[0][0]);

	TransformVertices(triangleVertices, screenSpaceVertices, 3, worldMatri, projectionMatri);


	for (uint32_t i = 0; i < 3; ++i)
	{
		glm::vec2 edge = screenSpaceVertices[(i + 2) % 3] - screenSpaceVertices[i];
		glm::vec2 pointToSide = pixel - glm::vec2{ screenSpaceVertices[i].x, screenSpaceVertices[i].y };

		if (Cross(edge, pointToSide) < 0)
		{
			return 0x00000000;	// Miss triangle
		}
	}

	return 0xffffffff;
}

__global__ void my_kernel(uint32_t* buf, uint32_t* depthBuffer)
{
	const int xPix = blockDim.x * blockIdx.x + threadIdx.x;
	const int yPix = blockDim.y * blockIdx.y + threadIdx.y;

	unsigned int pos = SCREEN_WIDTH * yPix + xPix;
	
	buf[pos] = getPixColor(xPix, yPix);
}

void gpuRender(uint32_t* buf, uint32_t* depthBuf) {
	const dim3 blocksPerGrid(H_TILES, V_TILES);
	const dim3 threadsPerBlock(TILE_WIDTH, TILE_HEIGHT);
	my_kernel<<<blocksPerGrid, threadsPerBlock>>>(buf, depthBuf);
}
