#include "RastizerDebugger.h"
#include "Camera.h"
#include "Texture.h"

RastizerDebugger::RastizerDebugger(Camera* pCamera, std::vector<Vertex_In>& vertices, std::vector<int> indices, Texture* pTexture)
	: m_pCamera{pCamera}
	, m_VerticesIn{vertices}
	, m_Indices{indices}
{
	m_VerticesOut.resize(vertices.size());
	m_Triangles.resize(vertices.size() / 3);

	m_DepthInfo = new depthInfo[SCREEN_WIDTH * SCREEN_HEIGHT];
	m_Buffer = new uint32_t[SCREEN_WIDTH * SCREEN_HEIGHT];

	m_pTexture = pTexture;
}

void RastizerDebugger::ClearDepthBuffer(depthInfo* depthBuf)
{
	for (int xPix = 0; xPix < SCREEN_WIDTH; ++xPix)
	{
		for (int yPix = 0; yPix < SCREEN_HEIGHT; ++yPix)
		{
			unsigned int pos = SCREEN_WIDTH * yPix + xPix;

			depthBuf[pos].depth = 1.0f;
		}
	}
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
	verteOutBuffer[index].normal = verteInBuffer[index].normal;
	verteOutBuffer[index].tangent = verteInBuffer[index].tangent;
	verteOutBuffer[index].color = verteInBuffer[index].color;
	verteOutBuffer[index].uv = verteInBuffer[index].uv;
}

void RastizerDebugger::FragmentShade(int xPix, int yPix, uint32_t* buf, depthInfo* depthBuf, const Triangle* primitives, int primitiveCount, TextureData* textures, int textureWidth, int textureHeight)
{
	unsigned int pos = SCREEN_WIDTH * yPix + xPix;

	glm::vec3 color{ 0.0f };
	//depthInfo depth{ 1.0f };
	for (int i = 0; i < primitiveCount; ++i)
	{
		if (xPix < primitives[i].boundingBox.min.x || xPix > primitives[i].boundingBox.max.x) continue;
		if (yPix < primitives[i].boundingBox.min.y || yPix > primitives[i].boundingBox.max.y) continue;

		if (!getPixColor(xPix, yPix, &depthBuf[pos], &color, primitives[i], textures, textureWidth, textureHeight))
			continue;
	}

	buf[pos] = (uint8_t)(color.b * 255.0f) | ((uint8_t)(color.g * 255) << 8) | ((uint8_t)(color.r * 255) << 16) | (uint8_t)(255.0f) << 24;
}

void RastizerDebugger::AssamblePrimitives(int index, int primitiveCount, const Vertex_Out* vertexBufferOut, Triangle* primitives, const int* bufIdx)
{
	if (index < primitiveCount)
	{
		for (int i = 0; i < 3; i++)
		{
			primitives[index].v[i] = vertexBufferOut[bufIdx[3 * index + i]];
		}

		primitives[index].boundingBox.min = glm::vec3{
			std::min(primitives[index].v[0].screenPosition.x, std::min(primitives[index].v[1].screenPosition.x, primitives[index].v[2].screenPosition.x)),
			std::min(primitives[index].v[0].screenPosition.y, std::min(primitives[index].v[1].screenPosition.y, primitives[index].v[2].screenPosition.y)),
			std::min(primitives[index].v[0].screenPosition.z, std::min(primitives[index].v[1].screenPosition.z, primitives[index].v[2].screenPosition.z)),
		};

		primitives[index].boundingBox.max = glm::vec3{
			std::max(primitives[index].v[0].screenPosition.x, std::max(primitives[index].v[1].screenPosition.x, primitives[index].v[2].screenPosition.x)),
			std::max(primitives[index].v[0].screenPosition.y, std::max(primitives[index].v[1].screenPosition.y, primitives[index].v[2].screenPosition.y)),
			std::max(primitives[index].v[0].screenPosition.z, std::max(primitives[index].v[1].screenPosition.z, primitives[index].v[2].screenPosition.z)),
		};

		//primitives[index].boundingBox = getAABBForTriangle(primitives[index]);
		primitives[index].visible = true;

	}
}

void RastizerDebugger::Render()
{

	int w = static_cast<int>(SCREEN_WIDTH);
	int h = static_cast<int>(SCREEN_HEIGHT);


	// Clear depth buffer
	ClearDepthBuffer(m_DepthInfo);

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
			FragmentShade(x,y, m_Buffer, m_DepthInfo, m_Triangles.data(), m_Triangles.size(), m_pTexture->GetData(), m_pTexture->GetWidth(), m_pTexture->GetHeight());
		}
	}

}

bool RastizerDebugger::getPixColor(int x, int y, depthInfo* pixelDepth, glm::vec3* color, Triangle primitive, TextureData* textures, int textureWidth, int textureHeight)
{
	float weights[3];
	float totalTriangleArea = abs(Cross(glm::vec2{ primitive.v[0].screenPosition } - glm::vec2{ primitive.v[2].screenPosition }, glm::vec2{ primitive.v[1].screenPosition } - glm::vec2{ primitive.v[2].screenPosition }));

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
			pixel - glm::vec2{ primitive.v[(i + 1) % 3].screenPosition },
			glm::vec2(glm::vec2{ primitive.v[(i + 2) % 3].screenPosition } - glm::vec2{ primitive.v[(i + 1) % 3].screenPosition })
		) / totalTriangleArea;
	}

	float currentDepth{};
	for (size_t i = 0; i < 3; ++i)
		currentDepth += (1.f / primitive.v[i].screenPosition.z) * weights[i];
	currentDepth = 1.f / currentDepth;

	if (pixelDepth[0].depth < currentDepth)
		return false;
	pixelDepth[0].depth = currentDepth;

	Vertex_Out endValue;
	float wInterpolated{};

	for (int i = 0; i < 3; ++i)
	{
		wInterpolated += (1.0f / primitive.v[i].screenPosition.w) * weights[i];

		endValue.normal += (primitive.v[i].normal) * weights[i];
		endValue.tangent += (primitive.v[i].tangent) * weights[i];
		endValue.color += primitive.v[i].color * weights[i];
		endValue.uv += (primitive.v[i].uv / primitive.v[i].screenPosition.w) * weights[i];
	}

	//endValue.color /= 3;
	endValue.uv *= (1.0f / wInterpolated);
	endValue.normal = glm::normalize((endValue.normal / 3.f));
	endValue.tangent = glm::normalize((endValue.tangent / 3.f));

	glm::vec3 endColor = TextureSample(textures, endValue.uv, textureWidth, textureHeight) * endValue.color;

	//lighting
	glm::vec3 lightDirection{ .577f, .577f, .577f };
	glm::vec3 lightColor{ 1.f,1.f,1.f };
	float intensity{ 2.f };

	// ambient
	glm::vec3 ambientColor{ 0.05f, 0.05f, 0.05f };

	float observedArea = std::max(0.0f, (glm::dot(endValue.normal, lightDirection)));

	glm::vec3 shadedEndColor{};

	shadedEndColor = lightColor * intensity * endColor * observedArea;
	shadedEndColor += ambientColor;

	color[0] = MaxToOne(shadedEndColor);
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
