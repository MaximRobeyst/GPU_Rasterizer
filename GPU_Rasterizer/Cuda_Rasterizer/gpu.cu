#include "gpu.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <device_launch_parameters.h>

#include "Camera.h"
#include "const.h"
#include "checkCUDAError.h"

#include "Texture.h"

#include <vector>
#include <iostream>

typedef unsigned char TextureData;

// https://github.com/bcrusco/CUDA-Rasterizer

static Camera* g_pCamera;

// Verte buffers
static Vertex_In* g_pVerteInBuffer = NULL;
static Vertex_Out* g_pVerteOutBuffer = NULL;
static int g_VertCount;

static TextureData* g_pTexture = NULL;

// inde buffer
static int* g_pIndeBuffer;
static int g_IndeCount;

// Triangle Buffer
static Triangle* dev_primitives = NULL;
static int primitiveCount = 0;

static depthInfo* g_DepthBuffer;

static int g_TextureWidth;
static int g_TextureHeight;

#define VERTBLOCKSIZE 256
#define FRAGBLOCKSIZE 256


void gpuInit(Camera* pCamera)
{
	g_pCamera = pCamera;
}

void InitBuffers(const std::vector<Vertex_In>& vertices,const std::vector<int>& indices, const std::vector<Texture*>& textures)
{
	g_VertCount = vertices.size();
	g_IndeCount = indices.size();
	primitiveCount = g_VertCount / 3;

	int screenSize = SCREEN_SIZE;

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

	err = cudaFree(g_DepthBuffer);
	if (err != cudaSuccess)
		std::cout << "error with freeing memoery vertexbuffer" << std::endl;
	err = cudaMalloc(&g_DepthBuffer, screenSize * sizeof(depthInfo));
	if (err != cudaSuccess)
		std::cout << "error with freeing memoery vertexbuffer" << std::endl;
	err = cudaMemset(g_DepthBuffer, 0, screenSize * sizeof(depthInfo));
	if (err != cudaSuccess)
		std::cout << "error with freeing memoery vertexbuffer" << std::endl;

	//err = cudaMemcpy(&g_DepthBuffer, depth.data(), SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
	//if (err != cudaSuccess)
	//	std::cout << "error with freeing memoery vertexbuffer" << std::endl;

	// Inde buffer
	cudaFree(g_pIndeBuffer);
	cudaMalloc(&g_pIndeBuffer, g_IndeCount * sizeof(int));
	cudaMemcpy(g_pIndeBuffer, indices.data(), sizeof(int) * g_IndeCount, cudaMemcpyHostToDevice);

	// Triangle buffer
	cudaFree(dev_primitives);
	cudaMalloc(&dev_primitives, g_VertCount / 3 * sizeof(Triangle));
	cudaMemset(dev_primitives, 0, g_VertCount / 3 * sizeof(Triangle));

	auto s = textures[0]->GetSize() * sizeof(TextureData);

	err = cudaFree(g_pTexture);
	err = cudaMalloc(&g_pTexture, s);
	err = cudaMemcpy(g_pTexture, &textures[0]->GetData()[0], s, cudaMemcpyHostToDevice);

	g_TextureWidth = textures[0]->GetWidth();
	g_TextureHeight = textures[0]->GetHeight();



	//texture<float, cudaTextureType2D,
	//	cudaReadModeElementType> texRef;
	//cudaChannelFormatDesc channelDesc =
	//	cudaCreateChannelDesc<float>();
	//size_t offset;
	//cudaBindTexture2D(&offset, texRef, devPtr, &channelDesc,
	//	width, height, pitch);



	checkCUDAError("initBuffers");
}


uint32_t * gpuAllocScreenBuffer(void)
{
	uint32_t* gpu_mem;

	cudaError_t err = cudaMalloc(&gpu_mem, SCREEN_SIZE * 4);
	if ( err != cudaSuccess ) return NULL;

	checkCUDAError("gpuAllocScreenBuffer");

	return gpu_mem;
};

float* gpuAllocDepthBuffer(void)
{
	float* depth_mem = NULL;

	cudaFree(depth_mem);
	cudaError_t err = cudaMalloc(&depth_mem, SCREEN_SIZE * sizeof(float));
	if (err != cudaSuccess) return NULL;

	cudaMemset(&depth_mem, 0, SCREEN_SIZE * sizeof(float));

	checkCUDAError("gpuAllocDepthBuffer");

	return depth_mem;
}

void gpuFree(void* gpu_mem) 
{
	cudaFree(gpu_mem);
}

int gpuBlit(void* src, void* dst)
{
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
	verteOutBuffer[index].normal = verteInBuffer[index].normal;
	verteOutBuffer[index].tangent = verteInBuffer[index].tangent;
	verteOutBuffer[index].color = verteInBuffer[index].color;
	verteOutBuffer[index].uv = verteInBuffer[index].uv;
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

		primitives[index].boundingBox.min = glm::vec3{
			min(primitives[index].v[0].screenPosition.x, min(primitives[index].v[1].screenPosition.x, primitives[index].v[2].screenPosition.x)),
			min(primitives[index].v[0].screenPosition.y, min(primitives[index].v[1].screenPosition.y, primitives[index].v[2].screenPosition.y)),
			min(primitives[index].v[0].screenPosition.z, min(primitives[index].v[1].screenPosition.z, primitives[index].v[2].screenPosition.z)),
		};

		primitives[index].boundingBox.max = glm::vec3{
			max(primitives[index].v[0].screenPosition.x, max(primitives[index].v[1].screenPosition.x, primitives[index].v[2].screenPosition.x)),
			max(primitives[index].v[0].screenPosition.y, max(primitives[index].v[1].screenPosition.y, primitives[index].v[2].screenPosition.y)),
			max(primitives[index].v[0].screenPosition.z, max(primitives[index].v[1].screenPosition.z, primitives[index].v[2].screenPosition.z)),
		};

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
glm::vec3 TextureSample(TextureData* textures, glm::vec2 uv, int width, int height)
{
	// Not bilinear
	int u = uv.x * width;
	int v = uv.y * height;

	// https://stackoverflow.com/questions/35005603/get-color-of-the-texture-at-uv-coordinate
	int uvIndex = 3 * (u + (v * width));

	// https://www.opengl.org/discussion_boards/showthread.php/170651-Is-it-possible-to-get-the-pixel-color
	float r = textures[uvIndex];
	float g = textures[uvIndex + 1];
	float b = textures[uvIndex + 2];

	glm::vec3 color = glm::vec3(r, g, b);
	color /= 255.f;

	return color;

	//return glm::vec3{ static_cast<float>(r) / 255.f,static_cast<float>(g) / 255.f,static_cast<float>(b) / 255.f };
}

__host__
__device__
bool getPixColor(int x, int y, depthInfo* pixelDepth, glm::vec3* color, Triangle primitive, TextureData* textures, int textureWidth, int textureHeight)
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

	if (pixelDepth[0].depth < currentDepth)
		return false;
	pixelDepth[0].depth = currentDepth;
	
	Vertex_Out endValue;
	float wInterpolated{};

	for (int i = 0; i < 3; ++i)
	{
		wInterpolated += (1.0f / primitive.v[i].screenPosition.w) * weights[i];

		endValue.normal += (primitive.v[i].normal) * weights[i];
		endValue.tangent += (primitive.v[i].tangent) * weights[i];
		endValue.color += primitive.v[i].color * weights[i];
		endValue.uv += (primitive.v[i].uv / primitive.v[i].screenPosition.w) * weights[i];
	}

	endValue.uv *= (1.0f / wInterpolated);
	endValue.normal = glm::normalize((endValue.normal / 3.f));
	endValue.tangent = glm::normalize((endValue.tangent / 3.f));

	glm::vec3 endColor = TextureSample(textures, endValue.uv, textureWidth, textureHeight) * endValue.color;

	//lighting
	glm::vec3 lightDirection{ -.577f, .577f, .577f };
	glm::vec3 lightColor{ 1.f,1.f,1.f };
	float intensity{ 2.f };

	// ambient
	glm::vec3 ambientColor{ 0.05f, 0.05f, 0.05f };
	
	float observedArea = max(0.0f, (glm::dot(endValue.normal, lightDirection)));
	
	glm::vec3 shadedEndColor{};

	shadedEndColor = lightColor * intensity * endColor * observedArea;
	shadedEndColor += ambientColor;

	color[0] = MaxToOne(shadedEndColor);
	return true;

}


__global__
void FragmentShading(uint32_t* buf, depthInfo* depthBuf, const Triangle* primitives, int primitiveCount, TextureData* textures, int textureWidth, int textureHeight)
{
	const int xPix = blockDim.x * blockIdx.x + threadIdx.x;
	const int yPix = blockDim.y * blockIdx.y + threadIdx.y;

	unsigned int pos = SCREEN_WIDTH * yPix + xPix;

	glm::vec3 color{ 0.0f };
	//depthInfo depth{ 1.0f };
	for (int i = 0; i < primitiveCount; ++i)
	{
		if (xPix < primitives[i].boundingBox.min.x || xPix > primitives[i].boundingBox.max.x) continue;
		if (yPix < primitives[i].boundingBox.min.y || yPix > primitives[i].boundingBox.max.y) continue;

		if (!getPixColor(xPix, yPix, &depthBuf[pos], &color, primitives[i], textures, textureWidth, textureHeight))
			continue;
	}

	buf[pos] = (uint8_t)(color.b * 255.0f) | ((uint8_t)(color.g * 255) << 8) | ((uint8_t)(color.r * 255) << 16) | (uint8_t)(255.0f) << 24;
}

__global__
void ClearDepthBuffer(depthInfo* depthBuf)
{
	const int xPix = blockDim.x * blockIdx.x + threadIdx.x;
	const int yPix = blockDim.y * blockIdx.y + threadIdx.y;

	unsigned int pos = SCREEN_WIDTH * yPix + xPix;

	depthBuf[pos].depth = 1.0f;
}

// Rasterizer loop
void gpuRender(uint32_t* buf) 
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
	ClearDepthBuffer << <blocksPerGrid, threadsPerBlock >> > (g_DepthBuffer);

	// Verte shading
	VerteShading<<<vertexGridSize, vertexBlockSize>>>(w, h,g_pCamera->GetFar(), g_pCamera->GetNear(), g_VertCount, g_pVerteInBuffer, g_pVerteOutBuffer, g_pCamera->GetViewMatrix(), g_pCamera->GetProjectionMatrix());

	// Primitive Assembly
	AssemblePrimitives<<<vertexGridSize, vertexBlockSize>>>(primitiveCount, g_pVerteOutBuffer, dev_primitives, g_pIndeBuffer);

	// Culling 


	// Rasterization
	FragmentShading<<<blocksPerGrid, threadsPerBlock >>>(buf, g_DepthBuffer, dev_primitives, primitiveCount, g_pTexture, g_TextureWidth, g_TextureHeight);


	checkCUDAError("gpuRender");
}