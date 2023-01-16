#include "RastizerDebugger.h"
#include "Camera.h"
#include "Texture.h"
#include "Mesh.h"
#include "Transform.h"

RastizerDebugger::RastizerDebugger(Camera* pCamera, std::vector<Vertex_In>& vertices, std::vector<unsigned int>& indices, Texture* pTexture)
	: m_pCamera{pCamera}
{
	m_DepthInfo = new int[SCREEN_WIDTH * SCREEN_HEIGHT];

	m_Fragments.resize(SCREEN_SIZE);
}

void RastizerDebugger::ClearDepthBuffer(int* depthBuf)
{
	m_Fragments.resize(SCREEN_SIZE);
	for (int xPix = 0; xPix < SCREEN_WIDTH; ++xPix)
	{
		for (int yPix = 0; yPix < SCREEN_HEIGHT; ++yPix)
		{
			unsigned int pos = SCREEN_WIDTH * yPix + xPix;

			depthBuf[pos] = INT_MAX;
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
			glm::clamp(std::max(primitives[index].v[0].screenPosition.x, std::max(primitives[index].v[1].screenPosition.x, primitives[index].v[2].screenPosition.x)), 0.0f, static_cast<float>(SCREEN_WIDTH)),
			glm::clamp(std::max(primitives[index].v[0].screenPosition.y, std::max(primitives[index].v[1].screenPosition.y, primitives[index].v[2].screenPosition.y)), 0.0f, static_cast<float>(SCREEN_HEIGHT)),
			std::max(primitives[index].v[0].screenPosition.z, std::max(primitives[index].v[1].screenPosition.z, primitives[index].v[2].screenPosition.z))
		};

		//primitives[index].boundingBox = getAABBForTriangle(primitives[index]);
		primitives[index].visible = true;

	}
}

void RastizerDebugger::Rasterize(int primId, Triangle* primitives, int primitveCount, Fragment* pFragmentBuffer, int* pDepthBuffer)
{
	int index = primId;

	if (index < primitveCount)
	{
		BoundingBox aabb = primitives[index].boundingBox;

		for (int xPixel = aabb.min.x; xPixel <= aabb.max.x; ++xPixel)
		{
			for (int yPixel = aabb.min.y; yPixel <= aabb.max.y; ++yPixel)
			{
				int depthIndex = yPixel * SCREEN_WIDTH + xPixel;

				glm::vec3 position;
				if (!PixelInTriangle(&primitives[index], glm::vec2{ xPixel, yPixel })) continue;
				int depthRepresentation = getDepthAtPixel(primitives[index]) * 1000.0f;

				pDepthBuffer[depthIndex] = std::min(pDepthBuffer[depthIndex], depthRepresentation);

				if (pDepthBuffer[depthIndex] == depthRepresentation)
				{
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
	}
}

void RastizerDebugger::InitBuffers(Camera* pCamera, std::vector<Vertex_In>& vertices, std::vector<unsigned int>& indices, Texture* pTexture, const glm::mat4& worldMatrix)
{
	m_VerticesIn.resize(vertices.size());
	std::copy(vertices.begin(), vertices.end(), m_VerticesIn.begin());

	m_Indices.resize(indices.size());
	std::copy(indices.begin(), indices.end(), m_Indices.begin());

	m_VerticesOut.resize(vertices.size());
	m_Triangles.resize(vertices.size() / 3);

	m_pTexture = pTexture;
}

void RastizerDebugger::Render(uint32_t* screen, std::vector<Mesh*>& meshes)
{
	for (int i = 0; i < meshes.size(); ++i)
	{
		InitBuffers(m_pCamera, meshes[i]->GetVertices(), meshes[i]->GetIndices(), meshes[i]->GetTextures().size() > 0 ? meshes[i]->GetTextures()[0] : nullptr, meshes[i]->GetTransform()->GetWorldTransform());
		Render(screen);
	}
}

void RastizerDebugger::ClearScreen(uint32_t* screen, glm::vec3 color)
{
	for (int x = 0; x < SCREEN_WIDTH; ++x)
	{
		for (int y = 0; y < SCREEN_HEIGHT; ++y)
		{
			unsigned int pos = SCREEN_WIDTH * y + x;
			screen[pos] = (uint8_t)(color.b * 255.0f) | ((uint8_t)(color.g * 255) << 8) | ((uint8_t)(color.r * 255) << 16) | (uint8_t)(255.0f) << 24;
		}
	}
}

void RastizerDebugger::Render(uint32_t* screen)
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
	for (int i = 0; i < m_Triangles.size(); ++i)
		Rasterize(i, m_Triangles.data(), m_Triangles.size(), m_Fragments.data(), m_DepthInfo);

	// Fragment shader
	for (int x = 0; x < SCREEN_WIDTH; ++x)
	{
		for (int y = 0; y < SCREEN_HEIGHT; ++y)
		{
			if (m_pTexture == nullptr)
				FragmentShade(x, y, screen, m_Fragments.data(), nullptr, 0, 0, 0);
			else
				FragmentShade(x,y, screen, m_Fragments.data(), m_pTexture->GetData(), m_pTexture->GetWidth(), m_pTexture->GetHeight(), m_pTexture->GetChannels());
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

	endValue.screenPosition /= 3.0f;
	endValue.uv *= (1.0f / wInterpolated);
	endValue.normal = glm::normalize((endValue.normal / 3.f));
	//endValue.tangent = glm::normalize((endValue.tangent / 3.f));

	return endValue;
}