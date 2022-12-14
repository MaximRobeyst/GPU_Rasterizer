#pragma once

#include <stdint.h>

#include "const.h"

#include <builtin_types.h>
#include <glm\glm.hpp>

#include "Camera.h"

void gpuInit();

uint32_t * gpuAlloc(void) ;
void gpuFree(void* gpu_mem);
int gpuBlit(void* src, void* dst);

void gpuRender(uint32_t* screenBuffer, uint32_t* depthBuffer);