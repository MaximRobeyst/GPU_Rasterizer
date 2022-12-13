#include "gpu.h"
#include <driver_types.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

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
	if ( err != cudaSuccess ) return 1;
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
uint32_t getPixColor(int x, int y) {

	glm::vec2 pixel = glm::vec2{ static_cast<float>(x), static_cast<float>(y) };
	glm::vec2 screenSpaceVertices[3];

	glm::vec3 triangleVertices[3]{
	glm::vec3{0.0f, 0.5f, -1.0f},
	glm::vec3{-0.5f, -0.5f, -1.0f},
	glm::vec3{0.5f, -0.5f, -1.0f}
	};


	for (uint32_t i = 0; i < 3; ++i)
	{
		screenSpaceVertices[i] = glm::vec2{
			((triangleVertices[i].x + 1.0f) / 2.0f) * SCREEN_WIDTH,
			((1.0f - triangleVertices[i].y) / 2.0f) * SCREEN_HEIGHT
		};
	}


	for (uint32_t i = 0; i < 3; ++i)
	{
		glm::vec2 edge = screenSpaceVertices[(i + 2) % 3] - screenSpaceVertices[i];
		glm::vec2 pointToSide = pixel - screenSpaceVertices[i];

		if (Cross(edge, pointToSide) < 0)
		{
			return 0x00000000;	// Miss triangle
		}
	}

	return 0xffffffff;
}

__global__ void my_kernel(uint32_t* buf) {
	const int xPix = blockDim.x * blockIdx.x + threadIdx.x;
	const int yPix = blockDim.y * blockIdx.y + threadIdx.y;

	unsigned int pos = SCREEN_WIDTH * yPix + xPix;
	
	buf[pos] = getPixColor(xPix, yPix);
}

void gpuRender(uint32_t* buf) {
	const dim3 blocksPerGrid(H_TILES, V_TILES);
	const dim3 threadsPerBlock(TILE_WIDTH, TILE_HEIGHT);
	my_kernel<<<blocksPerGrid, threadsPerBlock>>>(buf);
}
