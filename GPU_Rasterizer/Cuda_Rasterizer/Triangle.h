#pragma once
#include <glm\glm.hpp>
#include <cuda_runtime.h>

class Triangle final
{
public:
	Triangle(const glm::vec3& vertex0, const glm::vec3& vertex1, const glm::vec3& vertex2);

	__device__ __host__ void ProjectVertices(glm::vec2& topLeft, glm::vec2& bottomRight);

private:
	static const int TRIANGLE_POINTS = 3;

	glm::vec3 m_Vertices[TRIANGLE_POINTS];
	glm::vec4 m_ViewSpaceVertices[TRIANGLE_POINTS];
};