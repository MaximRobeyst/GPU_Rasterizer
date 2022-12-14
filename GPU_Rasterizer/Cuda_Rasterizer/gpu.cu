#include "gpu.h"
#include <driver_types.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "Camera.h"


__device__ Camera* g_pCamera;

void gpuInit()
{
	g_pCamera = new Camera{ glm::vec3{0,0,10.0f}, glm::vec3{0,0,1.0f}, 60.0f, SCREEN_WIDTH / SCREEN_HEIGHT };
}


uint32_t * gpuAlloc(void) {
	uint32_t* gpu_mem;

	cudaError_t err = cudaMalloc(&gpu_mem, SCREEN_SIZE * 4);
	if ( err != cudaSuccess ) return NULL;

	return gpu_mem;
};

void gpuFree(void* gpu_mem) {
	cudaFree(gpu_mem);
	delete g_pCamera;
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
void TransformVertex(const glm::vec3& originalVertex, glm::vec4& transformedVertex, glm::mat4 worldToView, glm::mat4 projectionMatrix)
{
	glm::vec4 projectedVertex = worldToView * glm::vec4(originalVertex, 1.0f);
	projectedVertex = projectionMatrix * projectedVertex;

	projectedVertex.x /= projectedVertex.w;
	projectedVertex.y /= projectedVertex.w;
	projectedVertex.z /= projectedVertex.w;

	projectedVertex.x = ((projectedVertex.x + 1) / 2) * SCREEN_WIDTH;
	projectedVertex.y = ((1 - projectedVertex.y) / 2) * SCREEN_HEIGHT;

	transformedVertex = projectedVertex;
}

__host__
__device__
void TransformVertices(const glm::vec3* originalVertex, glm::vec4* transformedVertex, int size, glm::mat4 worldToView, glm::mat4 projectionMatrix)
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

	glm::vec3 triangleVertices[3]{
		glm::vec3{0.0f, 2.0f, 0.0f},
		glm::vec3{-1.f, 0.f, 0.f},
		glm::vec3{1.f, 0.f, 0.0f}
	};

	TransformVertices(triangleVertices, screenSpaceVertices, 3, g_pCamera->GetWorldMatrix(), g_pCamera->GetProjectionMatrix());


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
