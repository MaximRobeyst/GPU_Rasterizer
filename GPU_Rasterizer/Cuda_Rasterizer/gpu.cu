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

__global__ 
void VerteShading(float far, float near, int verteCount, const Vertex_In* verteInBuffer, Vertex_Out* verteOutBuffer, const glm::mat4& worldViewProjectionMatri)
{
	int index = ((blockIdx.x * blockDim.x) + threadIdx.x) + (((blockIdx.y * blockDim.y) + threadIdx.y) * static_cast<int>(SCREEN_WIDTH));

	if (index >= verteCount) return;

	glm::vec4 projectedVertex = worldViewProjectionMatri * glm::vec4(verteInBuffer[index].position, 1.0f);
	glm::vec3 normDeviceCoordinates = glm::vec3(projectedVertex.x, projectedVertex.y, projectedVertex.z) / projectedVertex.w;

	verteOutBuffer[index].screenPosition =
		glm::vec4
	{
		static_cast<float>(SCREEN_WIDTH) * 0.5f * (normDeviceCoordinates.x + 1.0f),
		static_cast<float>(SCREEN_HEIGHT) * 0.5f * (normDeviceCoordinates.y + 1.0f),
		0.5f * ((far - near) * normDeviceCoordinates.z + (far + near)),
		projectedVertex.w
	};
	verteOutBuffer[index].color = verteInBuffer[index].color;
}

__host__ __device__ static
BoundingBox getAABBForTriangle(const Triangle tri) {
	BoundingBox boundingBox;
	boundingBox.min = glm::vec3(
		glm::min(glm::min(tri.v[0].screenPosition.x, tri.v[1].screenPosition.x), tri.v[2].screenPosition.x),
		glm::min(glm::min(tri.v[0].screenPosition.y, tri.v[1].screenPosition.y), tri.v[2].screenPosition.y),
		glm::min(glm::min(tri.v[0].screenPosition.z, tri.v[1].screenPosition.z), tri.v[2].screenPosition.z));
	boundingBox.max = glm::vec3(
		glm::max(glm::max(tri.v[0].screenPosition.x, tri.v[1].screenPosition.x), tri.v[2].screenPosition.x),
		glm::max(glm::max(tri.v[0].screenPosition.y, tri.v[1].screenPosition.y), tri.v[2].screenPosition.y),
		glm::max(glm::max(tri.v[0].screenPosition.z, tri.v[1].screenPosition.z), tri.v[2].screenPosition.z));
	return boundingBox;
}

__global__
void AssemblePrimitives(int primitiveCount, const Vertex_Out* vertexBufferOut, Triangle* primitives, const int* bufIdx) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < primitiveCount) {
		for (int i = 0; i < 3; i++) {
			primitives[index].v[i] = vertexBufferOut[bufIdx[3 * index + i]];
		}

		primitives[index].boundingBox = getAABBForTriangle(primitives[index]);
		primitives[index].visible = true;
	}
}

__host__ __device__ static
float calculateSignedArea(const Triangle tri) {
	return 0.5 * ((tri.v[2].screenPosition.x - tri.v[0].screenPosition.x) * (tri.v[1].screenPosition.y - tri.v[0].screenPosition.y) - (tri.v[1].screenPosition.x - tri.v[0].screenPosition.x) * (tri.v[2].screenPosition.y - tri.v[0].screenPosition.y));
}


__host__ __device__ static
float calculateBarycentricCoordinateValue(glm::vec2 a, glm::vec2 b, glm::vec2 c, const Triangle tri) {
	Triangle baryTri;
	baryTri.v[0].screenPosition = glm::vec4(a, 0, 0);
	baryTri.v[1].screenPosition = glm::vec4(b, 0, 0);
	baryTri.v[2].screenPosition = glm::vec4(c, 0, 0);
	return calculateSignedArea(baryTri) / calculateSignedArea(tri);
}

__host__ __device__ static
glm::vec3 calculateBarycentricCoordinate(const Triangle tri, glm::vec2 point) {
	float beta = calculateBarycentricCoordinateValue(glm::vec2(tri.v[0].screenPosition.x, tri.v[0].screenPosition.y), point, glm::vec2(tri.v[2].screenPosition.x, tri.v[2].screenPosition.y), tri);
	float gamma = calculateBarycentricCoordinateValue(glm::vec2(tri.v[0].screenPosition.x, tri.v[0].screenPosition.y), glm::vec2(tri.v[1].screenPosition.x, tri.v[1].screenPosition.y), point, tri);
	float alpha = 1.0 - beta - gamma;
	return glm::vec3(alpha, beta, gamma);
}

// CHECKITOUT
/**
 * Check if a barycentric coordinate is within the boundaries of a triangle.
 */
__host__ __device__ static
bool isBarycentricCoordInBounds(const glm::vec3 barycentricCoord) {
	return barycentricCoord.x >= 0.0 && barycentricCoord.x <= 1.0 &&
		barycentricCoord.y >= 0.0 && barycentricCoord.y <= 1.0 &&
		barycentricCoord.z >= 0.0 && barycentricCoord.z <= 1.0;
}

__host__ __device__ static
float getZAtCoordinate(const glm::vec3 barycentricCoord, const Triangle tri) {
	return -(barycentricCoord.x * tri.v[0].screenPosition.z
		+ barycentricCoord.y * tri.v[1].screenPosition.z
		+ barycentricCoord.z * tri.v[2].screenPosition.z);
}

/**
* Perform scanline rasterization on a triangle
*/
__global__
void rasterization(int w, int h, int primitiveCount, Triangle* primitives, uint32_t* buf) 
{
	int index = ((blockIdx.x * blockDim.x) + threadIdx.x) + (((blockIdx.y * blockDim.y) + threadIdx.y) * w);

	if (index < primitiveCount) 
	{
		// Only doing scanline triangle atm
		int minX = fmaxf(round(primitives[index].boundingBox.min.x), 0.0f), minY = fmaxf(round(primitives[index].boundingBox.min.y), 0.0f);
		int maxX = fminf(round(primitives[index].boundingBox.max.x), (float)w);

		// Loop through each scanline, then each pixel on the line
		for (int y = fminf(round(primitives[index].boundingBox.max.y), (float)h); y >= minY; y--) 
		{
			for (int x = minX; x <= maxX; x++) 
			{
				glm::vec3 baryCentricCoordinate = calculateBarycentricCoordinate(primitives[index], glm::vec2(x, y));
				if (isBarycentricCoordInBounds(baryCentricCoordinate)) 
				{
					int z = getZAtCoordinate(baryCentricCoordinate, primitives[index]) * 10000.0f;
					int depthIndex = w - x + (h - y) * w;
					// fill in value
				}
			}
		}
	}
}

__host__
__device__
uint32_t getPixColor(int x, int y) {
	return 0xffff0000;
}

// Rasterizer loop
void gpuRender(uint32_t* buf, uint32_t* depthBuf) {
	//const int xPix = blockDim.x * blockIdx.x + threadIdx.x;
	//const int yPix = blockDim.y * blockIdx.y + threadIdx.y;

	//unsigned int pos = SCREEN_WIDTH * yPix + xPix;

	int sideLength2d = 8;
	dim3 blockSize2d(sideLength2d, sideLength2d);
	dim3 blockCount2d((SCREEN_WIDTH + blockSize2d.x - 1) / blockSize2d.x,
		(SCREEN_HEIGHT + blockSize2d.y - 1) / blockSize2d.y);

	int vertexBlockSize = VERTBLOCKSIZE, fragmentBlockSize = FRAGBLOCKSIZE;
	int vertexGridSize = (g_VertCount + VERTBLOCKSIZE - 1) / VERTBLOCKSIZE;


	// Clear depth buffer

	// Verte shading
	VerteShading<<<vertexGridSize, vertexBlockSize>>>(g_pCamera->GetFar(), g_pCamera->GetNear(), g_VertCount, g_pVerteInBuffer, g_pVerteOutBuffer, g_pCamera->GetWorldViewProjectionMatrix());

	// Primitive Assembly
	//AssemblePrimitives<<<vertexGridSize, vertexBlockSize>>>(primitiveCount, g_pVerteOutBuffer, dev_primitives, g_pIndeBuffer);

	// Culling 


	// Rasterization
	//rasterization<<<blockCount2d, blockSize2d>>>(SCREEN_WIDTH, SCREEN_HEIGHT, primitiveCount, dev_primitives, buf);


	checkCUDAError("gpuRender");
	//buf[pos] = getPixColor(xPix, yPix);
}