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
	RastizerDebugger(Camera* pCamera, std::vector<Vertex_In>& vertices, std::vector<unsigned int>& indices, Texture* pTexture);


	void ClearDepthBuffer(int* depthBuf);
	void VertexShading(int index, int w, int h, float far, float near, int verteCount, const Vertex_In* verteInBuffer, Vertex_Out* verteOutBuffer, const glm::mat4 worldToView, const glm::mat4 projectionMatrix);
	void AssamblePrimitives(int inde,int primitiveCount, const Vertex_Out* vertexBufferOut, Triangle* primitives, const unsigned int* bufIdx);
	void Rasterize(int primId, Triangle* primitives, int primitveCount, Fragment* pFragmentBuffer, int* pDepthBuffer);
	void FragmentShade(int x, int y, uint32_t* buf, Fragment* pFragmentBuffer, TextureData* textures, int textureWidth, int textureHeight, int channels);

	void Render();
private:
	bool PixelInTriangle(Triangle* primitive, glm::vec2 pixel);
	float getDepthAtPixel(Triangle primitive);
	Fragment InterpolatePrimitiveValues(Triangle primitive);

	glm::vec3 MaxToOne(const glm::vec3 color);

	inline float Cross(const glm::vec2 v1, const glm::vec2 v2)
	{
		return v1.x * v2.y - v1.y * v2.x;
	}


	inline glm::vec3 TextureSample(TextureData* textures, glm::vec2 uv, int width, int height, int channels)
	{
		// Not bilinear
		int u = static_cast<int>(uv.x) * width;
		int v = static_cast<int>(uv.y) * height;

		// https://stackoverflow.com/questions/35005603/get-color-of-the-texture-at-uv-coordinate
		int uvIndex = channels * (u + (v * width));

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
	std::vector<Fragment> m_Fragments;
	std::vector<unsigned int> m_Indices;
	std::vector<Triangle> m_Triangles;
	
	int* m_DepthInfo{};
	uint32_t* m_Buffer{};
	Texture* m_pTexture;

	Camera* m_pCamera;
};

