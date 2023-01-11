#pragma once

#include <stdint.h>

#include "const.h"

#include <builtin_types.h>

#include <cuda.h>

#define GLM_FORCE_CUDA
#include <glm\glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>

class Camera;
class Texture;

typedef unsigned char TextureData;

struct Vertex_In
{
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec3 tangent;
	glm::vec3 color;
	glm::vec2 uv;
};

struct Vertex_Out
{
	glm::vec4 screenPosition;
	glm::vec3 normal;
	glm::vec3 tangent;
	glm::vec3 color;
	glm::vec2 uv;
};

struct depthInfo
{
	float depth;
};

struct BoundingBox {
	glm::vec3 min;
	glm::vec3 max;
};

struct Triangle {
	Vertex_Out v[3];
	BoundingBox boundingBox;
	bool visible;
};

void gpuInit(Camera* pCamera);
void InitBuffers(Vertex_In* vertices, int vertCount, const std::vector<unsigned int>& indices, const std::vector<Texture*>& textures, glm::mat4& worldMatrix);

void ClearDepthBuffer();
void ClearScreen(void* src);
void ClearScreen(uint32_t* src, glm::vec3 clearColor = glm::vec3{0.0f} );

uint32_t* gpuAllocScreenBuffer(void);
float * gpuAllocDepthBuffer(void) ;
void gpuFree(void* gpu_mem);
int gpuBlit(void* src, void* dst);

void gpuRender(uint32_t* screenBuffer);