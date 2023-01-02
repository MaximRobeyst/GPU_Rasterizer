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
void InitBuffers(const std::vector<Vertex_In>& vertices, const std::vector<int>& indices, const std::vector<Texture>& textures);

uint32_t* gpuAllocScreenBuffer(void);
float * gpuAllocDepthBuffer(void) ;
void gpuFree(void* gpu_mem);
int gpuBlit(void* src, void* dst);

void gpuRender(uint32_t* screenBuffer);