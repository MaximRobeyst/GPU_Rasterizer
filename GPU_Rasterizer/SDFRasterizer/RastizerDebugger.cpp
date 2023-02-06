#include "RastizerDebugger.h"
#include "Camera.h"
#include "Texture.h"
#include "Mesh.h"
#include "Transform.h"
#include <iostream>

RastizerDebugger::RastizerDebugger(Camera* pCamera, uint32_t* surface)
	: m_pCamera{pCamera}
{
	m_Buffer = surface;
	m_DepthInfo = new int[SCREEN_WIDTH * SCREEN_HEIGHT];
	m_pFragments = new Fragment[SCREEN_SIZE];
}

RastizerDebugger::~RastizerDebugger()
{
	delete m_DepthInfo;
	delete m_pFragments;
}

void RastizerDebugger::ClearDepthBuffer(int* depthBuf)
{
	for (int xPix = 0; xPix < SCREEN_WIDTH; ++xPix)
	{
		for (int yPix = 0; yPix < SCREEN_HEIGHT; ++yPix)
		{
			unsigned int pos = SCREEN_WIDTH * yPix + xPix;
			m_pFragments[pos] = Fragment{};
			depthBuf[pos] = INT_MAX;
		}
	}
}

void RastizerDebugger::VertexShading(int index, int w, int h, float, float, int verteCount, const Vertex_In* verteInBuffer, Vertex_Out* verteOutBuffer, const glm::mat4 worldToView, const glm::mat4 projectionMatrix, const glm::mat4& worldMatrix)
{
	if (index >= verteCount) return;

	glm::vec4 projectedVertex = projectionMatrix * worldToView * worldMatrix * glm::vec4(verteInBuffer[index].position, 1.0f);

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
	//verteOutBuffer[index].tangent = verteInBuffer[index].tangent;
	verteOutBuffer[index].color = verteInBuffer[index].color;
	verteOutBuffer[index].uv = verteInBuffer[index].uv;
}

void RastizerDebugger::AssamblePrimitives(int index, int primitiveCount, const Vertex_Out* vertexBufferOut, Triangle* primitives, const unsigned int* bufIdx)
{
	if (index < primitiveCount)
	{
		for (int i = 0; i < 3; i++)
		{
			primitives[index].v[i] = vertexBufferOut[bufIdx[3 * index + i]];
		}

		primitives[index].boundingBox.min = glm::vec3{
			glm::clamp(std::min(primitives[index].v[0].screenPosition.x, std::min(primitives[index].v[1].screenPosition.x, primitives[index].v[2].screenPosition.x)), 0.0f,static_cast<float>(SCREEN_WIDTH)),
			glm::clamp(std::min(primitives[index].v[0].screenPosition.y, std::min(primitives[index].v[1].screenPosition.y, primitives[index].v[2].screenPosition.y)), 0.0f, static_cast<float>(SCREEN_HEIGHT)),
			std::min(primitives[index].v[0].screenPosition.z, std::min(primitives[index].v[1].screenPosition.z, primitives[index].v[2].screenPosition.z))
		};

		primitives[index].boundingBox.max = glm::vec3{
			glm::clamp(std::max(primitives[index].v[0].screenPosition.x, std::max(primitives[index].v[1].screenPosition.x, primitives[index].v[2].screenPosition.x)), 0.0f, static_cast<float>(SCREEN_WIDTH-1)),
			glm::clamp(std::max(primitives[index].v[0].screenPosition.y, std::max(primitives[index].v[1].screenPosition.y, primitives[index].v[2].screenPosition.y)), 0.0f, static_cast<float>(SCREEN_HEIGHT)),
			std::max(primitives[index].v[0].screenPosition.z, std::max(primitives[index].v[1].screenPosition.z, primitives[index].v[2].screenPosition.z))
		};

		bool visible = true;
		if (primitives[index].boundingBox.max.x < 0) visible = false;
		if (primitives[index].boundingBox.max.y < 0) visible = false;

		if (primitives[index].boundingBox.min.x > SCREEN_WIDTH ) visible = false;
		if (primitives[index].boundingBox.min.y > SCREEN_HEIGHT) visible = false;

		if (primitives[index].boundingBox.max.z < 0.0f) visible = false;
		if (primitives[index].boundingBox.min.z > 1.0f) visible = false;

		primitives[index].visible = visible;

	}
}

void RastizerDebugger::Rasterize(int primId, Triangle* primitives, int primitveCount, Fragment* pFragmentBuffer, int* pDepthBuffer)
{
	int index = primId;

	if (index < primitveCount)
	{
		if (!primitives[index].visible) return;
		BoundingBox aabb = primitives[index].boundingBox;

		for (int xPixel = static_cast<int>(aabb.min.x); xPixel <=static_cast<int>(aabb.max.x); ++xPixel)
		{
			for (int yPixel = static_cast<int>(aabb.min.y); yPixel <= static_cast<int>(aabb.max.y); ++yPixel)
			{
				int depthIndex = yPixel * SCREEN_WIDTH + xPixel;

				glm::vec3 position;
				if (!PixelInTriangle(&primitives[index], glm::vec2{ xPixel, yPixel } )) continue;
				int depthRepresentation = static_cast<int>(getDepthAtPixel(primitives[index]) * INT_MAX);

				if (pDepthBuffer[depthIndex] > depthRepresentation)
				{
					pDepthBuffer[depthIndex] = depthRepresentation;
					pFragmentBuffer[depthIndex] = InterpolatePrimitiveValues(primitives[index]);
				}
			}
		}
	}
}

void RastizerDebugger::FragmentShade(int x, int y, uint32_t* buf, Fragment* pFragmentBuffer, TextureData* textures, int textureWidth, int textureHeight, int channels)
{
	unsigned int pos = SCREEN_WIDTH * y + x;

	if (pos < SCREEN_SIZE)
	{
		Fragment fragment = pFragmentBuffer[pos];
		if (fragment.screenPosition == glm::vec3{0.0f})
			return;

		glm::vec3 endColor = fragment.color;

		if (textures != nullptr)
			endColor = TextureSample(textures, fragment.uv, textureWidth, textureHeight, channels) * endColor;

		//lighting
		glm::vec3 lightDirection{ -.577f, .577f, .577f };
		glm::vec3 lightColor{ 1.f,1.f,1.f };
		float intensity{ 2.f };

		// ambient
		glm::vec3 ambientColor{ 0.05f, 0.05f, 0.05f };

		float observedArea = std::max(0.0f, (glm::dot(fragment.normal, lightDirection)));

		glm::vec3 shadedEndColor{};

		shadedEndColor = lightColor * intensity * endColor * observedArea;
		shadedEndColor += ambientColor;

		endColor = MaxToOne(shadedEndColor);
		buf[pos] = (uint8_t)(endColor.b * 255.0f) | ((uint8_t)(endColor.g * 255) << 8) | ((uint8_t)(endColor.r * 255) << 16) | (uint8_t)(255.0f) << 24;
		pFragmentBuffer[pos] = Fragment{};
	}
}

void RastizerDebugger::InitBuffers(Camera* /*pCamera*/, std::vector<Vertex_In>& vertices, std::vector<unsigned int>& indices, Texture* pTexture, const glm::mat4& worldMatrix)
{
	m_VerticesIn.resize(vertices.size());
	std::copy(vertices.begin(), vertices.end(), m_VerticesIn.begin());

	m_Indices.resize(indices.size());
	std::copy(indices.begin(), indices.end(), m_Indices.begin());

	m_VerticesOut.resize(vertices.size());
	m_Triangles.resize(vertices.size() / 3);

	m_pTexture = pTexture;

	m_WorldMatrix = worldMatrix;
}

void RastizerDebugger::Render(std::vector<Mesh*>& meshes)
{
	ClearScreen(glm::vec3{ 0.0f });
	for (int i = 0; i < meshes.size(); ++i)
	{
		InitBuffers(m_pCamera, meshes[i]->GetVertices(), meshes[i]->GetIndices(), meshes[i]->GetTextures().size() > 0 ? meshes[i]->GetTextures()[0] : nullptr, meshes[i]->GetTransform()->GetWorldTransform());
		Render();
	}
}

void RastizerDebugger::ClearScreen(glm::vec3 color)
{
	for (int x = 0; x < SCREEN_WIDTH; ++x)
	{
		for (int y = 0; y < SCREEN_HEIGHT; ++y)
		{
			unsigned int pos = SCREEN_WIDTH * y + x;
			m_Buffer[pos] = (uint8_t)(color.b * 255.0f) | ((uint8_t)(color.g * 255) << 8) | ((uint8_t)(color.r * 255) << 16) | (uint8_t)(255.0f) << 24;
		}
	}
}

void RastizerDebugger::ClearDepthBuffer()
{
	ClearDepthBuffer(m_DepthInfo);
}

void RastizerDebugger::Render()
{
	int w = static_cast<int>(SCREEN_WIDTH);
	int h = static_cast<int>(SCREEN_HEIGHT);

	// Verte shading
	for (int i = 0; i < m_VerticesIn.size(); ++i)
		VertexShading(i, w, h, m_pCamera->GetFar(), m_pCamera->GetNear(), static_cast<int>(m_VerticesIn.size()), m_VerticesIn.data(), m_VerticesOut.data(), m_pCamera->GetViewMatrix(), m_pCamera->GetProjectionMatrix(), m_WorldMatrix);

	// Primitive Assembly
	for(int i = 0; i < m_VerticesIn.size() / 3; ++i)
		AssamblePrimitives(i, static_cast<int>( m_VerticesIn.size() / 3), m_VerticesOut.data(), m_Triangles.data(), m_Indices.data());

	// Culling 


	// Rasterization
	for (int i = 0; i < m_Triangles.size(); ++i)
		Rasterize(i, m_Triangles.data(), static_cast<int>(m_Triangles.size()), m_pFragments, m_DepthInfo);

	// Fragment shader
	for (int y = 0; y < SCREEN_HEIGHT; ++y)
	{
		for (int x = 0; x < SCREEN_WIDTH; ++x)
		{
			FragmentShade(x,y, m_Buffer, m_pFragments, 
					m_pTexture == nullptr ? nullptr : m_pTexture->GetData(),
					m_pTexture == nullptr ? 0 : m_pTexture->GetWidth(),
					m_pTexture == nullptr ? 0 : m_pTexture->GetHeight(),
					m_pTexture == nullptr ? 0 : m_pTexture->GetChannels());
		}
	}

	

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

bool RastizerDebugger::PixelInTriangle(Triangle* primitive, glm::vec2 pixel)
{
	float totalTriangleArea = abs(Cross(glm::vec2{ primitive->v[0].screenPosition } - glm::vec2{ primitive->v[2].screenPosition }, glm::vec2{ primitive->v[1].screenPosition } - glm::vec2{ primitive->v[2].screenPosition }));

	for (int i = 0; i < 3; ++i)
	{
		if (primitive->v[i].screenPosition.z < 0.f || primitive->v[i].screenPosition.z > 1.f)
		{
			primitive->visible = false;
			return false;
		}
	}


	for (size_t i = 0; i < 3; ++i)
	{
		glm::vec2 p1{ primitive->v[(i + 2) % 3].screenPosition };
		glm::vec2 p2{ primitive->v[i].screenPosition };

		glm::vec2 edge = p1 - p2;
		glm::vec2 pointToSide = pixel - p2;
		if (Cross(edge, pointToSide) < 0)
		{
			primitive->visible = false;
			return false;
		}

		primitive->weights[i] = Cross(
			pixel - glm::vec2{ primitive->v[(i + 1) % 3].screenPosition },
			glm::vec2(glm::vec2{ primitive->v[(i + 2) % 3].screenPosition } - glm::vec2{ primitive->v[(i + 1) % 3].screenPosition })
		) / totalTriangleArea;
	}

	return true;
}

float RastizerDebugger::getDepthAtPixel(Triangle primitive)
{
	float currentDepth{};
	for (size_t i = 0; i < 3; ++i)
		currentDepth += (1.f / primitive.v[i].screenPosition.z) * primitive.weights[i];
	currentDepth = 1.f / currentDepth;

	return currentDepth;
}

__device__
Fragment RastizerDebugger::InterpolatePrimitiveValues(Triangle primitive)
{
	Fragment endValue;
	float wInterpolated{};

	for (int i = 0; i < 3; ++i)
	{
		wInterpolated += (1.0f / primitive.v[i].screenPosition.w) * primitive.weights[i];

		endValue.screenPosition += glm::vec3{ primitive.v[i].screenPosition } *primitive.weights[i];
		endValue.normal += (primitive.v[i].normal) * primitive.weights[i];
		//endValue.tangent += (primitive.v[i].tangent) * weights[i];
		endValue.color += primitive.v[i].color * primitive.weights[i];
		endValue.uv += (primitive.v[i].uv / primitive.v[i].screenPosition.w) * primitive.weights[i];
	}

	assert(wInterpolated != 0);

	endValue.screenPosition /= 3.0f;
	endValue.uv *= (1.0f / wInterpolated);
	endValue.normal = glm::normalize((endValue.normal / 3.f));
	//endValue.tangent = glm::normalize((endValue.tangent / 3.f));

	return endValue;
}