#include "Triangle.h"
#include "const.h"

#include <algorithm>

float Clamp(float value, float min, float max)
{
	if (value < min)
		return min;
	if (value > max)
		return max;

	return value;
}

__device__ 
__host__
void Triangle::ProjectVertices(glm::vec2& topLeft, glm::vec2& bottomRight)
{
	for (size_t i = 0; i < TRIANGLE_POINTS; i++)
	{
		glm::vec4 transformedVertex{};
		//SceneGraph::GetInstance()->GetCamera()->VertexTransformationFunction(m_Vertices[i].position, transformedVertex, screenWidth, screenHeight);
		m_ViewSpaceVertices[i] = transformedVertex;
	}

	topLeft.x = std::min({ m_ViewSpaceVertices[0].x, m_ViewSpaceVertices[1].x, m_ViewSpaceVertices[2].x });
	bottomRight.x = std::max({ m_ViewSpaceVertices[0].x, m_ViewSpaceVertices[1].x, m_ViewSpaceVertices[2].x });

	topLeft.y = std::min({ m_ViewSpaceVertices[0].y, m_ViewSpaceVertices[1].y, m_ViewSpaceVertices[2].y });
	bottomRight.y = std::max({ m_ViewSpaceVertices[0].y, m_ViewSpaceVertices[1].y, m_ViewSpaceVertices[2].y });

	topLeft.x = Clamp(topLeft.x, 0.f, SCREEN_WIDTH - 1.f);
	bottomRight.x =  Clamp(bottomRight.x, 0.f, SCREEN_WIDTH - 1.f);

	topLeft.y = Clamp(topLeft.y, 0.f, SCREEN_HEIGHT - 1.f);
	bottomRight.y = Clamp(bottomRight.y, 0.f, SCREEN_HEIGHT - 1.f);
}
