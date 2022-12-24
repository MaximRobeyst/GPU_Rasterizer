#include "RastizerDebugger.h"
#include "Camera.h"

RastizerDebugger::RastizerDebugger(Camera* pCamera, std::vector<Vertex_In>& vertices, std::vector<int> indices)
	: m_pCamera{pCamera}
	, m_VerticesIn{vertices}
	, m_Indices{indices}
{
	m_VerticesOut.resize(vertices.size());
	m_Triangles.resize(vertices.size() / 3);
}

void RastizerDebugger::VertexShading(int index, int w, int h, float, float, int verteCount, const Vertex_In* verteInBuffer, Vertex_Out* verteOutBuffer, const glm::mat4 worldToView, const glm::mat4 projectionMatrix)
{
	if (index >= verteCount) return;

	glm::vec4 projectedVertex = projectionMatrix * worldToView * glm::vec4(verteInBuffer[index].position, 1.0f);

	glm::vec3 normDeviceCoordinates = glm::vec3(projectedVertex.x, projectedVertex.y, projectedVertex.z) / projectedVertex.w;

	verteOutBuffer[index].screenPosition =
		glm::vec4
	{
		((normDeviceCoordinates.x + 1) / 2) * w,
		((1 - normDeviceCoordinates.y) / 2) * h,
		normDeviceCoordinates.z,
		projectedVertex.w
	};
	verteOutBuffer[index].color = verteInBuffer[index].color;
}

void RastizerDebugger::FragmentShade(int xPix, int yPix, Triangle* primitives, int primitiveCount)
{
	unsigned int pos = SCREEN_WIDTH * yPix + xPix;

	glm::vec3 color{ 0.0f };
	float depth = 1.0f;
	for (int i = 0; i < primitiveCount; ++i)
	{
		if (!getPixColor(xPix, yPix, &depth, &color, primitives[i]))
			continue;
	}

}

void RastizerDebugger::AssamblePrimitives(int index, int primitiveCount, const Vertex_Out* vertexBufferOut, Triangle* primitives, const int* bufIdx)
{
	if (index < primitiveCount)
	{
		for (int i = 0; i < 3; i++)
		{
			primitives[index].v[i] = vertexBufferOut[bufIdx[3 * index + i]];
		}

		//primitives[index].boundingBox = getAABBForTriangle(primitives[index]);
		primitives[index].visible = true;
	}
}

void RastizerDebugger::Render()
{

	int w = static_cast<int>(SCREEN_WIDTH);
	int h = static_cast<int>(SCREEN_HEIGHT);


	// Clear depth buffer
	//ClearDepthBuffer(depthBuf);

	// Verte shading
	for (int i = 0; i < m_VerticesIn.size(); ++i)
		VertexShading(i, w, h, m_pCamera->GetFar(), m_pCamera->GetNear(), m_VerticesIn.size(), m_VerticesIn.data(), m_VerticesOut.data(), m_pCamera->GetViewMatrix(), m_pCamera->GetProjectionMatrix());

	// Primitive Assembly
	for(int i = 0; i < m_VerticesIn.size() / 3; ++i)
		AssamblePrimitives(i, m_VerticesIn.size() / 3, m_VerticesOut.data(), m_Triangles.data(), m_Indices.data());

	// Culling 


	// Rasterization
	for (int x = 0; x < SCREEN_WIDTH; ++x)
	{
		for (int y = 0; y < SCREEN_HEIGHT; ++y)
		{
			FragmentShade(x,y, m_Triangles.data(), m_Triangles.size());
		}
	}

}

bool RastizerDebugger::getPixColor(int x, int y, float* pixelDepth, glm::vec3* color, Triangle primitive)
{
	float weights[3];

	glm::vec2 screenPositions[3]{ glm::vec2{primitive.v[0].screenPosition}, glm::vec2{primitive.v[1].screenPosition}, glm::vec2{primitive.v[2].screenPosition} };

	float totalTriangleArea = abs(Cross(screenPositions[0] - screenPositions[2], screenPositions[1] - screenPositions[2]));

	glm::vec2 pixel{ x, y };

	for (int i = 0; i < 3; ++i)
	{
		if (primitive.v[i].screenPosition.z < 0.f || primitive.v[i].screenPosition.z > 1.f) return false;
		//else if (p.x > screenWidth || p.y > screenHeight || p.x < 0 || p.y < 0) return false;
	}

	for (size_t i = 0; i < 3; ++i)
	{
		glm::vec2 p1{ primitive.v[(i + 2) % 3].screenPosition };
		glm::vec2 p2{ primitive.v[i].screenPosition };

		glm::vec2 edge = p1 - p2;
		glm::vec2 pointToSide = pixel - p2;
		if (Cross(edge, pointToSide) < 0)
			return false;

		weights[i] = Cross(
			pixel - screenPositions[(i + 1) % 3],
			(screenPositions[(i + 2) % 3] - screenPositions[(i + 1) % 3])
		) / totalTriangleArea;
	}

	float currentDepth{};
	for (size_t i = 0; i < 3; ++i)
		currentDepth += (1.f / primitive.v[i].screenPosition.z) * weights[i];
	currentDepth = 1.f / currentDepth;

	if (pixelDepth[0] < currentDepth)
		return false;
	pixelDepth[0] = currentDepth;

	Vertex_Out endValue;

	for (int i = 0; i < 3; ++i)
	{
		endValue.color += primitive.v[i].color;
	}

	endValue.color /= 3;

	color[0] = MaxToOne(endValue.color);
	return true;
}

glm::vec3 RastizerDebugger::MaxToOne(const glm::vec3 color)
{
	float highestValue{ 1 };

	if (color.r > highestValue)
		highestValue = color.r;
	if (color.g > highestValue)
		highestValue = color.g;
	if (color.b > highestValue)
		highestValue = color.b;

	return glm::vec3{ color.r / highestValue, color.g / highestValue, color.b / highestValue };
}
