#pragma once
#include <host_defines.h>

#include "nvMatrix.h"
#include <vector_types.h>

typedef nv::matrix4<float> float4x4;

class Camera
{
public:
	__host__ __device__ Camera(float3 position, float3 forward, float FOV, float far, float near);

	__device__ float4x4 GetWorldViewProjectionMatri();
	__device__ float4x4 GetViewProjectionMatri();
	__device__ float4x4 GetWorldMatri();

private:
	float4x4 m_ViewProjection;
	float4x4 m_World;
};

