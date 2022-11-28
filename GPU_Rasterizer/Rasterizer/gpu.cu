#include "gpu.h"
#include <driver_types.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <vector>
#include <math.h>

#include "Triangle.h"
#include "helper_math.h"

uint32_t* gpuAlloc(void) {
	uint32_t* gpu_mem;

	cudaError_t err = cudaMalloc(&gpu_mem, SCREEN_SIZE * 4);
	if (err != cudaSuccess) return NULL;

	return gpu_mem;
};

void gpuFree(void* gpu_mem) {
	cudaFree(gpu_mem);
}

int gpuBlit(void* src, void* dst) {
	cudaError_t err = cudaMemcpy(dst, src, SCREEN_SIZE * 4, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) return 1;
	return 0;
}

__host__
__device__
uint32_t getPixColor(int x, int y) {
	Triangle g_Triangle{ float3{0.0f, 0.5f, -1.0f}, float3{-0.5, -0.5f, -1.0f}, float3{0.5f, -0.5f, -1.0f} };

	float4 color{0.0f, 0.0f, 0.0f, 0.0f};

	float3 pixel{ static_cast<float>(x), static_cast<float>(y), 0.0f };

	// Triangle
	float3 a = make_float3(g_Triangle.m_ViewSpaceVertices[1]) - make_float3(g_Triangle.m_ViewSpaceVertices[0]);
	float3 b = make_float3(g_Triangle.m_ViewSpaceVertices[2]) - make_float3(g_Triangle.m_ViewSpaceVertices[0]);

	auto normal = cross(a, b);

	if (dot(normal, pixel) == 0)
		color = float4{ 1.0f, 1.0f, 1.0f, 1.0f };

	return (uint8_t)(color.x * 255) | ((uint8_t)(color.y * 255) << 8) | ((uint8_t)(color.z * 255) << 16) | ((uint8_t)(color.w * 255) << 24);
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
	my_kernel <<<blocksPerGrid, threadsPerBlock>>> (buf);
}