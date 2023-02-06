#include "Mesh.h"

#include "Material.h"
#include "Texture.h"
#include "Transform.h"
#include "Camera.h"

bool Mesh::m_Lines{ false };

Mesh::Mesh(std::vector<Vertex_In>& vertices, std::vector<unsigned int>& indices)
	: m_Vertices{vertices}
	, m_Indices{indices}
{
}

Mesh::~Mesh()
{
	delete m_pTransform;

	for (auto pTexture : m_pTextures)
	{
		delete pTexture;
	}

	for (auto pMesh :  m_pSubmeshes)
	{
		delete pMesh;
	}
	m_pTextures.clear();
	m_pSubmeshes.clear();
}

Mesh::Mesh(const Mesh& mesh)
	: m_Vertices{mesh.m_Vertices}
	, m_Indices{mesh.m_Indices}
{
}

void Mesh::SetMaterial(Material* pMaterial)
{
	m_pMaterial = pMaterial;
}

void Mesh::AddTexture(Texture* pTexture)
{
	m_pTextures.emplace_back(pTexture);
}

void Mesh::SetTransform(Transform* pTransform)
{
	m_pTransform = pTransform;
}

Material* Mesh::GetMaterial() const
{
	return m_pMaterial;
}

Transform* Mesh::GetTransform() const
{
	return m_pTransform;
}

size_t Mesh::GetSubmeshCount() const
{
	return m_pSubmeshes.size();
}

Mesh* Mesh::GetSubmesh(int id) const
{
	return m_pSubmeshes[id];
}

void Mesh::ReadyTextures()
{
}

void Mesh::SetLinesRender(bool lines)
{
	m_Lines = lines;
}

bool Mesh::GetLinesRender()
{
	return m_Lines;
}

std::vector<Vertex_In>& Mesh::GetVertices()
{
	return m_Vertices;
}

std::vector<unsigned int>& Mesh::GetIndices()
{
	return m_Indices;
}

std::vector<Texture*>& Mesh::GetTextures()
{
	return m_pTextures;
}

Vertex_In* Mesh::GetVertexPointer()
{
	return m_Vertices.data();
}

const unsigned int* Mesh::GetIndexPointer() const
{
	return m_Indices.data();
}

Mesh* Mesh::CreateCube(float width, float height, float depth)
{
	std::vector<Vertex_In> vertices;
	std::vector<unsigned int> indices;

	float halfWidth = width / 2.0f;
	float halfHeight = height / 2.0f;
	float halfDepth = depth / 2.0f;

	// Create Top Face
	indices.emplace_back(static_cast<unsigned int>(vertices.size()) + 0);
	indices.emplace_back(static_cast<unsigned int>(vertices.size()) + 1);
	indices.emplace_back(static_cast<unsigned int>(vertices.size()) + 3);

	indices.emplace_back(static_cast<unsigned int>(vertices.size()) + 1);
	indices.emplace_back(static_cast<unsigned int>(vertices.size()) + 2);
	indices.emplace_back(static_cast<unsigned int>(vertices.size()) + 3);

	vertices.emplace_back(Vertex_In{ glm::vec3{-halfWidth,	halfHeight,	-halfDepth},	glm::normalize(glm::vec3{0,1,0}) , glm::vec3{1,1,1}, glm::vec2{1.0f, 1.f} });
	vertices.emplace_back(Vertex_In{ glm::vec3{halfWidth,	halfHeight,	-halfDepth},		glm::normalize(glm::vec3{0,1,0}) , glm::vec3{1,1,1}, glm::vec2{1.0f, 0.f} });
	vertices.emplace_back(Vertex_In{ glm::vec3{halfWidth,	halfHeight,	halfDepth},			glm::normalize(glm::vec3{0,1,0}) , glm::vec3{1,1,1}, glm::vec2{0.0f, 0.f} });
	vertices.emplace_back(Vertex_In{ glm::vec3{-halfWidth,	halfHeight,	halfDepth},		glm::normalize(glm::vec3{0,1,0}) , glm::vec3{1,1,1}, glm::vec2{0.0f, 1.f} });

	// Create Bottom face
	indices.emplace_back(static_cast<unsigned int>(vertices.size()) + 0);
	indices.emplace_back(static_cast<unsigned int>(vertices.size()) + 1);
	indices.emplace_back(static_cast<unsigned int>(vertices.size()) + 3);

	indices.emplace_back(static_cast<unsigned int>(vertices.size()) + 1);
	indices.emplace_back(static_cast<unsigned int>(vertices.size()) + 2);
	indices.emplace_back(static_cast<unsigned int>(vertices.size()) + 3);


	vertices.emplace_back(Vertex_In{ glm::vec3{-halfWidth,	-halfHeight,	-halfDepth},	glm::normalize(glm::vec3{0,-1,0}), glm::vec3{1,1,1}, glm::vec2{1.0f, 1.f} });
	vertices.emplace_back(Vertex_In{ glm::vec3{halfWidth,	-halfHeight,	-halfDepth},		glm::normalize(glm::vec3{0,-1,0}), glm::vec3{1,1,1}, glm::vec2{1.0f, 0.f} });
	vertices.emplace_back(Vertex_In{ glm::vec3{halfWidth,	-halfHeight,	halfDepth},			glm::normalize(glm::vec3{0,-1,0}), glm::vec3{1,1,1}, glm::vec2{0.0f, 0.f} });
	vertices.emplace_back(Vertex_In{ glm::vec3{-halfWidth,	-halfHeight,	halfDepth},		glm::normalize(glm::vec3{0,-1,0}), glm::vec3{1,1,1}, glm::vec2{0.0f, 1.f} });

	// Left
	indices.emplace_back(static_cast<unsigned int>(vertices.size()) + 0);
	indices.emplace_back(static_cast<unsigned int>(vertices.size()) + 1);
	indices.emplace_back(static_cast<unsigned int>(vertices.size()) + 3);

	indices.emplace_back(static_cast<unsigned int>(vertices.size()) + 1);
	indices.emplace_back(static_cast<unsigned int>(vertices.size()) + 2);
	indices.emplace_back(static_cast<unsigned int>(vertices.size()) + 3);

	vertices.emplace_back(Vertex_In{ glm::vec3{-halfWidth,	halfHeight,	halfDepth},		glm::normalize(glm::vec3{-1,0,0}), glm::vec3{1,1,1}, glm::vec2{1.0f, 1.f} });
	vertices.emplace_back(Vertex_In{ glm::vec3{-halfWidth,	-halfHeight,	halfDepth},		glm::normalize(glm::vec3{-1,0,0}), glm::vec3{1,1,1}, glm::vec2{1.0f, 0.f} });
	vertices.emplace_back(Vertex_In{ glm::vec3{-halfWidth,	-halfHeight,	-halfDepth},	glm::normalize(glm::vec3{-1,0,0}), glm::vec3{1,1,1}, glm::vec2{0.0f, 0.f} });
	vertices.emplace_back(Vertex_In{ glm::vec3{-halfWidth,	halfHeight,	-halfDepth},	glm::normalize(glm::vec3{-1,0,0}), glm::vec3{1,1,1}, glm::vec2{0.0f, 1.f} });

	// Right
	indices.emplace_back(static_cast<unsigned int>(vertices.size()) + 0);
	indices.emplace_back(static_cast<unsigned int>(vertices.size()) + 1);
	indices.emplace_back(static_cast<unsigned int>(vertices.size()) + 3);

	indices.emplace_back(static_cast<unsigned int>(vertices.size()) + 1);
	indices.emplace_back(static_cast<unsigned int>(vertices.size()) + 2);
	indices.emplace_back(static_cast<unsigned int>(vertices.size()) + 3);

	vertices.emplace_back(Vertex_In{ glm::vec3{halfWidth,	halfHeight,	halfDepth},		glm::normalize(glm::vec3{1,0,0}), glm::vec3{1,1,1}, glm::vec2{1.0f, 1.f} });
	vertices.emplace_back(Vertex_In{ glm::vec3{halfWidth,	-halfHeight,	halfDepth},		glm::normalize(glm::vec3{1,0,0}), glm::vec3{1,1,1}, glm::vec2{1.0f, 0.f} });
	vertices.emplace_back(Vertex_In{ glm::vec3{halfWidth,	-halfHeight,	-halfDepth},	glm::normalize(glm::vec3{1,0,0}), glm::vec3{1,1,1}, glm::vec2{0.0f, 0.f} });
	vertices.emplace_back(Vertex_In{ glm::vec3{halfWidth,	halfHeight,	-halfDepth},	glm::normalize(glm::vec3{1,0,0}), glm::vec3{1,1,1}, glm::vec2{0.0f, 1.f} });


	// Front
	indices.emplace_back(static_cast<unsigned int>(vertices.size()) + 0);
	indices.emplace_back(static_cast<unsigned int>(vertices.size()) + 1);
	indices.emplace_back(static_cast<unsigned int>(vertices.size()) + 3);

	indices.emplace_back(static_cast<unsigned int>(vertices.size()) + 1);
	indices.emplace_back(static_cast<unsigned int>(vertices.size()) + 2);
	indices.emplace_back(static_cast<unsigned int>(vertices.size()) + 3);

	vertices.emplace_back(Vertex_In{ glm::vec3{halfWidth,	halfHeight,	halfDepth},		glm::normalize(glm::vec3{0,0,1})	,glm::vec3{1,1,1}, glm::vec2{1.0f, 1.f} });
	vertices.emplace_back(Vertex_In{ glm::vec3{halfWidth,	-halfHeight,	halfDepth},		glm::normalize(glm::vec3{0,0,1})	,glm::vec3{1,1,1}, glm::vec2{1.0f, 0.f} });
	vertices.emplace_back(Vertex_In{ glm::vec3{-halfWidth,	-halfHeight,	halfDepth}, glm::normalize(glm::vec3{0,0,1})	,glm::vec3{1,1,1}, glm::vec2{0.0f, 0.f} });
	vertices.emplace_back(Vertex_In{ glm::vec3{-halfWidth,	halfHeight,	halfDepth}, glm::normalize(glm::vec3{0,0,1})	,glm::vec3{1,1,1}, glm::vec2{0.0f, 1.f} });

	// Back
	indices.emplace_back(static_cast<unsigned int>(vertices.size()) + 0);
	indices.emplace_back(static_cast<unsigned int>(vertices.size()) + 1);
	indices.emplace_back(static_cast<unsigned int>(vertices.size()) + 3);

	indices.emplace_back(static_cast<unsigned int>(vertices.size()) + 1);
	indices.emplace_back(static_cast<unsigned int>(vertices.size()) + 2);
	indices.emplace_back(static_cast<unsigned int>(vertices.size()) + 3);

	vertices.emplace_back(Vertex_In{ glm::vec3{halfWidth,	halfHeight,	-halfDepth},		glm::normalize(glm::vec3{0,0,-1}) , glm::vec3{1,1,1}, glm::vec2{1.0f, 1.f} });
	vertices.emplace_back(Vertex_In{ glm::vec3{halfWidth,	-halfHeight,	-halfDepth},		glm::normalize(glm::vec3{0,0,-1}) , glm::vec3{1,1,1}, glm::vec2{1.0f, 0.f} });
	vertices.emplace_back(Vertex_In{ glm::vec3{-halfWidth,	-halfHeight,	-halfDepth},	glm::normalize(glm::vec3{0,0,-1}) , glm::vec3{1,1,1}, glm::vec2{0.0f, 0.f} });
	vertices.emplace_back(Vertex_In{ glm::vec3{-halfWidth,	halfHeight,	-halfDepth},	glm::normalize(glm::vec3{0,0,-1}) , glm::vec3{1,1,1}, 	glm::vec2{0.0f, 1.f} });

	return new Mesh(vertices, indices);
}
