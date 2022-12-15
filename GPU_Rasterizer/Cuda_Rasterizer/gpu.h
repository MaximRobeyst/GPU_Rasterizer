#pragma once

#include <stdint.h>

#include "const.h"

#include <builtin_types.h>

#include <cuda.h>

#define GLM_FORCE_CUDA
#include <glm\glm.hpp>
#include <glm/gtc/type_ptr.hpp>

class Camera;

void gpuInit(Camera* pCamera);

uint32_t * gpuAlloc(void) ;
void gpuFree(void* gpu_mem);
int gpuBlit(void* src, void* dst);

void gpuRender(uint32_t* screenBuffer, uint32_t* depthBuffer);