#pragma once

#define GLM_FORCE_CUDA
#include <glm\glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "gpu.h"

class Camera;

class RastizerDebugger
{
public:
	RastizerDebugger(Camera* pCamera, std::vector<Vertex_In>& vertices, std::vector<int> indices);

	void VertexShading(int index, int w, int h, float far, float near, int verteCount, const Vertex_In* verteInBuffer, Vertex_Out* verteOutBuffer, const glm::mat4 worldToView, const glm::mat4 projectionMatrix);
	void AssamblePrimitives(int inde,int primitiveCount, const Vertex_Out* vertexBufferOut, Triangle* primitives, const int* bufIdx);
	void FragmentShade(int xPix, int yPos, Triangle* primitives, int primitiveCount);

	void Render();

	bool getPixColor(int x, int y, float* pixelDepth, glm::vec3* color, Triangle primitive);

private:
	glm::vec3 MaxToOne(const glm::vec3 color);

	inline float Cross(const glm::vec2 v1, const glm::vec2 v2)
	{
		return v1.x * v2.y - v1.y * v2.x;
	}


	std::vector<Vertex_In> m_VerticesIn;
	std::vector<Vertex_Out> m_VerticesOut;
	std::vector<int> m_Indices;
	std::vector<Triangle> m_Triangles;
	
	Camera* m_pCamera;
};

