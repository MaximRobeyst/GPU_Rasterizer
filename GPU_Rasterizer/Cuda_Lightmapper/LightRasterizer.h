#pragma once
#include <vector>

#include <glm\glm.hpp>
#include "Mesh.h"

enum class ProjectionType
{
	XY_Projection,
	XZ_Projection,
	YZ_Projection
};

class LightRasterizer
{
public:
	void MapVerticesToCoords(Vertex v0, Vertex v1, Vertex v2);

private:
	void XYProjection(glm::vec2& lighmapCoords, Vertex& vertex);
	void XZProjection(glm::vec2& lighmapCoords, Vertex& vertex);
	void YZProjection(glm::vec2& lighmapCoords, Vertex& vertex);

	std::vector<glm::vec2> m_LightmapCoords;
	int m_Lightmaps = 0;

	ProjectionType m_ProjectionType;
};