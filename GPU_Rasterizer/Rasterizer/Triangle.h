#pragma once
#include <builtin_types.h>

#include "const.h"
#include <iostream>

class Triangle
{
public:
	__host__ __device__ Triangle(const float3& vert0, const float3& vert1, const float3& vert2);

	float3 m_Vertices[3];
	float4 m_ViewSpaceVertices[3];

};

