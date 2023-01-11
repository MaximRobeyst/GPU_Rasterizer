#pragma once

#define GLM_FORCE_CUDA
#pragma warning(push, 0)
#include <glm\glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#pragma warning(pop)

#include "gpu.h"

class Camera;

class RastizerDebugger
{
public:
	RastizerDebugger(Camera* pCamera, std::vector<Vertex_In>& vertices, std::vector<int> indices, Texture* pTexture);


	void ClearDepthBuffer(depthInfo* depthBuf);
	void VertexShading(int index, int w, int h, float far, float near, int verteCount, const Vertex_In* verteInBuffer, Vertex_Out* verteOutBuffer, const glm::mat4 worldToView, const glm::mat4 projectionMatrix);
	void AssamblePrimitives(int inde,int primitiveCount, const Vertex_Out* vertexBufferOut, Triangle* primitives, const int* bufIdx);
	void FragmentShade(int xPix, int yPix, uint32_t* buf, depthInfo* depthBuf, const Triangle* primitives, int primitiveCount, TextureData* textures, int textureWidth, int textureHeight);

	void Render();

	bool getPixColor(int x, int y, depthInfo* pixelDepth, glm::vec3* color, Triangle primitive, TextureData* textures, int textureWidth, int textureHeight);

private:
	glm::vec3 MaxToOne(const glm::vec3 color);

	inline float Cross(const glm::vec2 v1, const glm::vec2 v2)
	{
		return v1.x * v2.y - v1.y * v2.x;
	}


	inline glm::vec3 TextureSample(TextureData* textures, glm::vec2 uv, int width, int height)
	{
		// Not bilinear
		int u = static_cast<int>(uv.x) * width;
		int v = static_cast<int>(uv.y) * height;

		// https://stackoverflow.com/questions/35005603/get-color-of-the-texture-at-uv-coordinate
		int uvIndex = 3 * (u + (v * width));

		// https://www.opengl.org/discussion_boards/showthread.php/170651-Is-it-possible-to-get-the-pixel-color
		float r = textures[uvIndex];
		float g = textures[uvIndex + 1];
		float b = textures[uvIndex + 2];

		glm::vec3 color = glm::vec3(r, g, b);
		color /= 255.f;

		return color;

		//return glm::vec3{ static_cast<float>(r) / 255.f,static_cast<float>(g) / 255.f,static_cast<float>(b) / 255.f };
	}


	std::vector<Vertex_In> m_VerticesIn;
	std::vector<Vertex_Out> m_VerticesOut;
	std::vector<int> m_Indices;
	std::vector<Triangle> m_Triangles;
	
	depthInfo* m_DepthInfo{};
	uint32_t* m_Buffer{};
	Texture* m_pTexture;

	Camera* m_pCamera;
};

