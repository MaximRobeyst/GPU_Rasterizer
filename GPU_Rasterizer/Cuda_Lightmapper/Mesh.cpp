#include "Mesh.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "Material.h"
#include "Texture.h"
#include "Transform.h"
#include "Camera.h"

bool Mesh::m_Lines{ false };

Mesh::Mesh(std::vector<Vertex>& vertices, std::vector<unsigned int>& indices)
	: m_Vertices{vertices}
	, m_Indices{indices}
{
}

Mesh::~Mesh()
{
	glDeleteVertexArrays(1, &m_VAO);
	glDeleteBuffers(1, &m_VBO);
	glDeleteBuffers(1, &m_EBO);

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

void Mesh::Initialize()
{
	glGenVertexArrays(1, &m_VAO);
	glGenBuffers(1, &m_VBO);
	glGenBuffers(1, &m_EBO);
	// bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
	glBindVertexArray(m_VAO);

	glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
	glBufferData(GL_ARRAY_BUFFER, m_Vertices.size() * sizeof(Vertex), m_Vertices.data(), GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_Indices.size() * sizeof(unsigned int), m_Indices.data(), GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(sizeof(glm::vec3)));
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(sizeof(glm::vec3) * 2));
	glEnableVertexAttribArray(2);

	// note that this is allowed, the call to glVertexAttribPointer registered VBO as the vertex attribute's bound vertex buffer object so afterwards we can safely unbind
	//glBindBuffer(GL_ARRAY_BUFFER, 0);

	// remember: do NOT unbind the EBO while a VAO is active as the bound element buffer object IS stored in the VAO; keep the EBO bound.
	//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	// You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
	// VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
	//glBindVertexArray(0);
}

void Mesh::Draw(Camera* pCamera)
{
	for (auto pMesh : m_pSubmeshes)
	{
		pMesh->Draw(pCamera);
	}

	for (int i = 0; i < m_pTextures.size(); ++i)
	{
		glActiveTexture(GL_TEXTURE0 + i);
		glBindTexture(GL_TEXTURE_2D, m_pTextures[i]->GetTextureID());
	}

	m_pMaterial->UseProgram();

	m_pMaterial->SetMat4("model", m_pTransform->GetWorldTransform());
	m_pMaterial->SetMat4("view", pCamera->GetViewMatrix());
	m_pMaterial->SetMat4("projection", pCamera->GetProjectionMatrix());


	//if (Mesh::GetLinesRender())
	//	glPolygonMode(GL_FRONT_AND_BACK, GL_LINES);
	//else
	//	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glBindVertexArray(m_VAO); // seeing as we only have a single VAO there's no need to bind it every time, but we'll do so to keep things a bit more organized
	//glDrawArrays(GL_TRIANGLES, 0, 6);
	glDrawElements(m_Lines ?  GL_LINES : GL_TRIANGLES, m_Indices.size(), GL_UNSIGNED_INT, 0);
	// glBindVertexArray(0); // no need to unbind it every time 
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

void Mesh::AddSubmesh(Mesh* pMesh)
{
	m_pSubmeshes.emplace_back(pMesh);
	pMesh->Initialize();
}

Material* Mesh::GetMaterial() const
{
	return m_pMaterial;
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
	for (int i = 0; i < m_pSubmeshes.size(); ++i)
	{
		m_pSubmeshes[i]->ReadyTextures();
	}

	m_pMaterial->UseProgram();
	m_pMaterial->SetInt("texture1", 0);
	m_pMaterial->SetInt("texture2", 1);
}

void Mesh::SetLinesRender(bool lines)
{
	m_Lines = lines;
}

bool Mesh::GetLinesRender()
{
	return m_Lines;
}

const std::vector<Vertex>& Mesh::GetVertices() const
{
	return m_Vertices;
}

const std::vector<unsigned int>& Mesh::GetIndices() const
{
	return m_Indices;
}

Mesh* Mesh::CreateCube(float width, float height, float depth)
{
	std::vector<Vertex> vertices;
	std::vector<unsigned int> indices;

	float halfWidth = width / 2.0f;
	float halfHeight = height / 2.0f;
	float halfDepth = depth / 2.0f;

	// Create Top Face
	indices.emplace_back(vertices.size() + 0);
	indices.emplace_back(vertices.size() + 1);
	indices.emplace_back(vertices.size() + 3);

	indices.emplace_back(vertices.size() + 1);
	indices.emplace_back(vertices.size() + 2);
	indices.emplace_back(vertices.size() + 3);

	vertices.emplace_back(Vertex{ glm::vec3{-halfWidth,	halfHeight,	-halfDepth},	glm::normalize(glm::vec3{0,1,0}) ,glm::vec2{1.0f, 1.f} });
	vertices.emplace_back(Vertex{ glm::vec3{halfWidth,	halfHeight,	-halfDepth},	glm::normalize(glm::vec3{0,1,0}) ,glm::vec2{1.0f, 0.f} });
	vertices.emplace_back(Vertex{ glm::vec3{halfWidth,	halfHeight,	halfDepth},	glm::normalize(glm::vec3{0,1,0}) ,glm::vec2{0.0f, 0.f} });
	vertices.emplace_back(Vertex{ glm::vec3{-halfWidth,	halfHeight,	halfDepth},	glm::normalize(glm::vec3{0,1,0}) ,glm::vec2{0.0f, 1.f} });

	// Create Bottom face
	indices.emplace_back(vertices.size() + 0);
	indices.emplace_back(vertices.size() + 1);
	indices.emplace_back(vertices.size() + 3);

	indices.emplace_back(vertices.size() + 1);
	indices.emplace_back(vertices.size() + 2);
	indices.emplace_back(vertices.size() + 3);

	vertices.emplace_back(Vertex{ glm::vec3{-halfWidth,	-halfHeight,	-halfDepth},	glm::normalize(glm::vec3{0,-1,0}) ,glm::vec2{1.0f, 1.f} });
	vertices.emplace_back(Vertex{ glm::vec3{halfWidth,	-halfHeight,	-halfDepth},	glm::normalize(glm::vec3{0,-1,0}) ,glm::vec2{1.0f, 0.f} });
	vertices.emplace_back(Vertex{ glm::vec3{halfWidth,	-halfHeight,	halfDepth},	glm::normalize(glm::vec3{0,-1,0}) ,glm::vec2{0.0f, 0.f} });
	vertices.emplace_back(Vertex{ glm::vec3{-halfWidth,	-halfHeight,	halfDepth},	glm::normalize(glm::vec3{0,-1,0}) ,glm::vec2{0.0f, 1.f} });

	// Left
	indices.emplace_back(vertices.size() + 0);
	indices.emplace_back(vertices.size() + 1);
	indices.emplace_back(vertices.size() + 3);

	indices.emplace_back(vertices.size() + 1);
	indices.emplace_back(vertices.size() + 2);
	indices.emplace_back(vertices.size() + 3);

	vertices.emplace_back(Vertex{ glm::vec3{-halfWidth,	halfHeight,	halfDepth},	glm::normalize(glm::vec3{-1,0,0}) ,		glm::vec2{1.0f, 1.f} });
	vertices.emplace_back(Vertex{ glm::vec3{-halfWidth,	-halfHeight,	halfDepth},	glm::normalize(glm::vec3{-1,0,0}) ,	glm::vec2{1.0f, 0.f} });
	vertices.emplace_back(Vertex{ glm::vec3{-halfWidth,	-halfHeight,	-halfDepth},	glm::normalize(glm::vec3{-1,0,0}) ,	glm::vec2{0.0f, 0.f} });
	vertices.emplace_back(Vertex{ glm::vec3{-halfWidth,	halfHeight,	-halfDepth},	glm::normalize(glm::vec3{-1,0,0}) ,		glm::vec2{0.0f, 1.f} });

	// Right
	indices.emplace_back(vertices.size() + 0);
	indices.emplace_back(vertices.size() + 1);
	indices.emplace_back(vertices.size() + 3);

	indices.emplace_back(vertices.size() + 1);
	indices.emplace_back(vertices.size() + 2);
	indices.emplace_back(vertices.size() + 3);

	vertices.emplace_back(Vertex{ glm::vec3{halfWidth,	halfHeight,	halfDepth},	glm::normalize(glm::vec3{1,0,0}) ,	 glm::vec2{1.0f, 1.f} });
	vertices.emplace_back(Vertex{ glm::vec3{halfWidth,	-halfHeight,	halfDepth},	glm::normalize(glm::vec3{1,0,0}) ,glm::vec2{1.0f, 0.f} });
	vertices.emplace_back(Vertex{ glm::vec3{halfWidth,	-halfHeight,	-halfDepth},	glm::normalize(glm::vec3{1,0,0}) ,glm::vec2{0.0f, 0.f} });
	vertices.emplace_back(Vertex{ glm::vec3{halfWidth,	halfHeight,	-halfDepth},	glm::normalize(glm::vec3{1,0,0})		,glm::vec2{0.0f, 1.f} });


	// Front
	indices.emplace_back(vertices.size() + 0);
	indices.emplace_back(vertices.size() + 1);
	indices.emplace_back(vertices.size() + 3);

	indices.emplace_back(vertices.size() + 1);
	indices.emplace_back(vertices.size() + 2);
	indices.emplace_back(vertices.size() + 3);

	vertices.emplace_back(Vertex{ glm::vec3{halfWidth,	halfHeight,	halfDepth}, glm::normalize(glm::vec3{0,0,1})	,	 glm::vec2{1.0f, 1.f} });
	vertices.emplace_back(Vertex{ glm::vec3{halfWidth,	-halfHeight,	halfDepth}, glm::normalize(glm::vec3{0,0,1})	,glm::vec2{1.0f, 0.f} });
	vertices.emplace_back(Vertex{ glm::vec3{-halfWidth,	-halfHeight,	halfDepth}, glm::normalize(glm::vec3{0,0,1})	,glm::vec2{0.0f, 0.f} });
	vertices.emplace_back(Vertex{ glm::vec3{-halfWidth,	halfHeight,	halfDepth}, glm::normalize(glm::vec3{0,0,1})	,	 glm::vec2{0.0f, 1.f} });

	// Back
	indices.emplace_back(vertices.size() + 0);
	indices.emplace_back(vertices.size() + 1);
	indices.emplace_back(vertices.size() + 3);

	indices.emplace_back(vertices.size() + 1);
	indices.emplace_back(vertices.size() + 2);
	indices.emplace_back(vertices.size() + 3);

	vertices.emplace_back(Vertex{ glm::vec3{halfWidth,	halfHeight,	-halfDepth},	glm::normalize(glm::vec3{0,0,-1}) ,		glm::vec2{1.0f, 1.f} });
	vertices.emplace_back(Vertex{ glm::vec3{halfWidth,	-halfHeight,	-halfDepth},	glm::normalize(glm::vec3{0,0,-1}) ,	glm::vec2{1.0f, 0.f} });
	vertices.emplace_back(Vertex{ glm::vec3{-halfWidth,	-halfHeight,	-halfDepth},	glm::normalize(glm::vec3{0,0,-1}) ,	glm::vec2{0.0f, 0.f} });
	vertices.emplace_back(Vertex{ glm::vec3{-halfWidth,	halfHeight,	-halfDepth},	glm::normalize(glm::vec3{0,0,-1}) ,		glm::vec2{0.0f, 1.f} });

	return new Mesh(vertices, indices);
}
