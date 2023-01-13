
#include <stdint.h>
#include <builtin_types.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <device_launch_parameters.h>

#include "Mesh.h"
#include "lightMapper.cuh"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "LightRasterizer.h"

#include <algorithm>

#define VERTBLOCKSIZE 256
#define FRAGBLOCKSIZE 256

void LightRasterizer::MapVerticesToCoords(std::vector<Vertex>& vertices)
{
	const auto maxX = std::max_element(vertices.begin(), vertices.end(), [](Vertex lhs, Vertex rhs) {
			return lhs.position.x < rhs.position.x;
		})->position.x;

	const auto maxY = std::max_element(vertices.begin(), vertices.end(), [](Vertex lhs, Vertex rhs) {
		return lhs.position.x < rhs.position.x;
		})->position.y;

	const auto maxZ = std::max_element(vertices.begin(), vertices.end(), [](Vertex lhs, Vertex rhs) {
		return lhs.position.x < rhs.position.x;
		})->position.z;

	auto maxAxis = std::max(maxX, maxY, maxZ);
	if (maxAxis == maxX) m_ProjectionType = ProjectionType::XY_Projection;
	if (maxAxis == maxY) m_ProjectionType = ProjectionType::XZ_Projection;
	if (maxAxis == maxZ) m_ProjectionType = ProjectionType::YZ_Projection;

	m_LightmapCoords.resize(vertices.size());
	for (int i = 0; i < m_LightmapCoords.size(); ++i)
	{
		switch (m_ProjectionType)
		{
		case ProjectionType::XY_Projection:
			XYProjection(m_LightmapCoords[i], vertices[i]);
			break;
		case ProjectionType::XZ_Projection:
			XZProjection(m_LightmapCoords[i], vertices[i]);
			break;
		case ProjectionType::YZ_Projection:
			YZProjection(m_LightmapCoords[i], vertices[i]);
			break;
		}
	}


}

void LightRasterizer::XYProjection(glm::vec2& lighmapCoords, Vertex& vertex)
{
	lighmapCoords.x = vertex.position.x;
	lighmapCoords.y = vertex.position.y;
}

void LightRasterizer::XZProjection(glm::vec2& lighmapCoords, Vertex& vertex)
{
	lighmapCoords.x = vertex.position.x;
	lighmapCoords.y = vertex.position.z;
}

void LightRasterizer::YZProjection(glm::vec2& lighmapCoords, Vertex& vertex)
{
	lighmapCoords.x = vertex.position.y;
	lighmapCoords.y = vertex.position.z;
}
