#include "gpu.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <device_launch_parameters.h>

#include "Camera.h"
#include "const.h"
#include "checkCUDAError.h"

#include <vector>
#include <iostream>

// https://github.com/bcrusco/CUDA-Rasterizer

static Camera* g_pCamera;

// Verte buffers
static Vertex_In* g_pVerteInBuffer = NULL;
static Vertex_Out* g_pVerteOutBuffer = NULL;
static int g_VertCount;

// inde buffer
static int* g_pIndeBuffer;
static int g_IndeCount;

// Triangle Buffer
static Triangle* dev_primitives = NULL;
static int primitiveCount = 0;

static float* g_DepthBuffer;

#define VERTBLOCKSIZE 256
#define FRAGBLOCKSIZE 256


void gpuInit(Camera* pCamera)
{
	g_pCamera = pCamera;
}

void InitBuffers(const std::vector<Vertex_In>& vertices,const std::vector<int>& indices)
{
	g_VertCount = vertices.size();
	g_IndeCount = indices.size();
	primitiveCount = g_VertCount / 3;

	// Verte in buffer
	cudaError_t err = cudaFree(g_pVerteInBuffer);
	if (err != cudaSuccess)
		std::cout << "error with freeing memoery vertexbuffer" << std::endl;

	err = cudaMalloc(&g_pVerteInBuffer, g_VertCount * sizeof(Vertex_In));
	if (err != cudaSuccess)
		std::cout << "error with freeing memoery vertexbuffer" << std::endl;
	err = cudaMemcpy(g_pVerteInBuffer, vertices.data(), g_VertCount * sizeof(Vertex_In), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
		std::cout << "error with freeing memoery vertexbuffer" << std::endl;

	// verte out buffer
	err = cudaFree(g_pVerteOutBuffer);
	if (err != cudaSuccess)
		std::cout << "error with freeing memoery vertexbuffer" << std::endl;
	err = cudaMalloc(&g_pVerteOutBuffer, g_VertCount * sizeof(Vertex_Out));
	if (err != cudaSuccess)
		std::cout << "error with freeing memoery vertexbuffer" << std::endl;
	err = cudaMemset(g_pVerteOutBuffer, 0, g_VertCount * sizeof(Vertex_In));
	if (err != cudaSuccess)
		std::cout << "error with freeing memoery vertexbuffer" << std::endl;

	// Inde buffer
	cudaFree(g_pIndeBuffer);
	cudaMalloc(&g_pIndeBuffer, g_IndeCount * sizeof(int));
	cudaMemcpy(g_pIndeBuffer, indices.data(), sizeof(int) * g_IndeCount, cudaMemcpyHostToDevice);

	// Triangle buffer
	cudaFree(dev_primitives);
	cudaMalloc(&dev_primitives, g_VertCount / 3 * sizeof(Triangle));
	cudaMemset(dev_primitives, 0, g_VertCount / 3 * sizeof(Triangle));

	checkCUDAError("initBuffers");
}


uint32_t * gpuAllocScreenBuffer(void)
{
	uint32_t* gpu_mem;

	cudaError_t err = cudaMalloc(&gpu_mem, SCREEN_SIZE * 4);
	if ( err != cudaSuccess ) return NULL;

	return gpu_mem;
};

float* gpuAllocDepthBuffer(void)
{
	float* gpu_mem;

	cudaError_t err = cudaMalloc(&gpu_mem, SCREEN_SIZE * sizeof(float));
	if (err != cudaSuccess) return NULL;

	return gpu_mem;
}

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
inline float Cross(const glm::vec2 v1, const glm::vec2 v2)
{
	return v1.x * v2.y - v1.y * v2.x;
}

__global__ 
void VerteShading(int w, int h, float far, float near, int verteCount, const Vertex_In* verteInBuffer, Vertex_Out* verteOutBuffer, const glm::mat4 worldToView ,const glm::mat4 projectionMatrix)
{
	int index = ((blockIdx.x * blockDim.x) + threadIdx.x) + (((blockIdx.y * blockDim.y) + threadIdx.y) * w);

	if (index >= verteCount) return;

	glm::vec4 projectedVertex = projectionMatrix * worldToView * glm::vec4(verteInBuffer[index].position, 1.0f);

	glm::vec3 normDeviceCoordinates = glm::vec3(projectedVertex.x, projectedVertex.y, projectedVertex.z) / projectedVertex.w;

	verteOutBuffer[index].screenPosition =
		glm::vec4
	{
		((normDeviceCoordinates.x + 1) / 2) * w,
		((1 - normDeviceCoordinates.y) / 2) * h,
		normDeviceCoordinates.z,
		projectedVertex.w
	};
	verteOutBuffer[index].color = verteInBuffer[index].color;
}

__global__
void AssemblePrimitives(int primitiveCount, const Vertex_Out* vertexBufferOut, Triangle* primitives, const int* bufIdx) 
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < primitiveCount) 
	{
		for (int i = 0; i < 3; i++) 
		{
			primitives[index].v[i] = vertexBufferOut[bufIdx[3 * index + i]];
		}

		//primitives[index].boundingBox = getAABBForTriangle(primitives[index]);
		primitives[index].visible = true;
	}
}

__host__
__device__
glm::vec3 MaxToOne(const glm::vec3 color)
{
	float highestValue{ 1 };

	if (color.r > highestValue)
		highestValue = color.r;
	if (color.g > highestValue)
		highestValue = color.g;
	if (color.b > highestValue)
		highestValue = color.b;

	return glm::vec3{ color.r / highestValue, color.g / highestValue, color.b / highestValue };
}

__host__
__device__
bool getPixColor(int x, int y, float* pixelDepth, glm::vec3* color, Triangle primitive)
{
	float weights[3];
	float totalTriangleArea = abs(Cross(glm::vec2{ primitive.v[0].screenPosition } - glm::vec2{ primitive.v[2].screenPosition }, glm::vec2{ primitive.v[1].screenPosition } - glm::vec2{ primitive.v[2].screenPosition }));

	glm::vec2 pixel{ x, y };

	for (int i = 0; i < 3; ++i)
	{
		if (primitive.v[i].screenPosition.z < 0.f || primitive.v[i].screenPosition.z > 1.f) return false;
		//else if (p.x > screenWidth || p.y > screenHeight || p.x < 0 || p.y < 0) return false;
	}

	for (size_t i = 0; i < 3; ++i)
	{
		glm::vec2 p1{ primitive.v[(i + 2) % 3].screenPosition };
		glm::vec2 p2{ primitive.v[i].screenPosition };

		glm::vec2 edge = p1 - p2;
		glm::vec2 pointToSide = pixel - p2;
		if (Cross(edge, pointToSide) < 0)
			return false;

		weights[i] = Cross(
			pixel - glm::vec2{ primitive.v[(i + 1) % 3].screenPosition },
			glm::vec2(glm::vec2{ primitive.v[(i + 2) % 3].screenPosition } - glm::vec2{ primitive.v[(i + 1) % 3].screenPosition })
		) / totalTriangleArea;
	}

	float currentDepth{};
	for (size_t i = 0; i < 3; ++i)
		currentDepth += (1.f / primitive.v[i].screenPosition.z) * weights[i];
	currentDepth = 1.f / currentDepth;

	if (pixelDepth[0] < currentDepth)
		return false;
	pixelDepth[0] = currentDepth;

	Vertex_Out endValue;

	for (int i = 0; i < 3; ++i)
	{
		endValue.color += primitive.v[i].color * weights[i];
	}

	endValue.color /= 3;

	color[0] = MaxToOne(endValue.color);
	return true;
}

__global__
void FragmentShading(uint32_t* buf, float* depthBuf, const Triangle* primitives, int primitiveCount)
{
	const int xPix = blockDim.x * blockIdx.x + threadIdx.x;
	const int yPix = blockDim.y * blockIdx.y + threadIdx.y;

	unsigned int pos = SCREEN_WIDTH * yPix + xPix;

	glm::vec3 color{ 0.0f };
	float depth = 1.0f;
	for (int i = 0; i < primitiveCount; ++i)
	{
		if (!getPixColor(xPix, yPix, &depth, &color, primitives[i]))
			continue;
	}

	buf[pos] = (uint8_t)(color.b * 255.0f) | ((uint8_t)(color.g * 255) << 8) | ((uint8_t)(color.r * 255) << 16) | (uint8_t)(255.0f) << 24;
}

void ClearDepthBuffer(float* depthBuf)
{
	for(int i = 0; i < SCREEN_SIZE; ++i)
		depthBuf[i] = 1.f;
}

// Rasterizer loop
void gpuRender(uint32_t* buf, float* depthBuf) 
{
	int sideLength2d = 8;
	dim3 blockSize2d(sideLength2d, sideLength2d);
	dim3 blockCount2d((SCREEN_WIDTH + blockSize2d.x - 1) / blockSize2d.x,
		(SCREEN_HEIGHT + blockSize2d.y - 1) / blockSize2d.y);

	int vertexBlockSize = VERTBLOCKSIZE, fragmentBlockSize = FRAGBLOCKSIZE;
	int vertexGridSize = (g_VertCount + VERTBLOCKSIZE - 1) / VERTBLOCKSIZE;

	const dim3 blocksPerGrid(H_TILES, V_TILES);
	const dim3 threadsPerBlock(TILE_WIDTH, TILE_HEIGHT);

	int w = static_cast<int>(SCREEN_WIDTH);
	int h = static_cast<int>(SCREEN_HEIGHT);


	// Clear depth buffer
	//ClearDepthBuffer(depthBuf);

	// Verte shading
	VerteShading<<<vertexGridSize, vertexBlockSize>>>(w, h,g_pCamera->GetFar(), g_pCamera->GetNear(), g_VertCount, g_pVerteInBuffer, g_pVerteOutBuffer, g_pCamera->GetViewMatrix(), g_pCamera->GetProjectionMatrix());

	// Primitive Assembly
	AssemblePrimitives<<<vertexGridSize, vertexBlockSize>>>(primitiveCount, g_pVerteOutBuffer, dev_primitives, g_pIndeBuffer);

	// Culling 


	// Rasterization
	FragmentShading<<<blocksPerGrid, threadsPerBlock >>>(buf, depthBuf, dev_primitives, primitiveCount);


	checkCUDAError("gpuRender");
	//buf[pos] = getPixColor(xPix, yPix);
}