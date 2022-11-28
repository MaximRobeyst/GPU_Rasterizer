#include "Triangle.h"

Triangle::Triangle(const float3& vert0, const float3& vert1, const float3& vert2)
{
	m_Vertices[0] = vert0;
	m_Vertices[1] = vert1;
	m_Vertices[2] = vert2;

	for (int i = 0; i < 3; ++i)
	{
		m_ViewSpaceVertices[i].x = ((m_Vertices[i].x + 1) / 2) * static_cast<float>(SCREEN_WIDTH);
		m_ViewSpaceVertices[i].y = ((1 - m_Vertices[i].y) / 2) * static_cast<float>(SCREEN_HEIGHT);

		m_ViewSpaceVertices[i].z = 0;
	}
}
