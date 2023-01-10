#pragma once
#include <vector>
#include <glm/glm.hpp>

#include "gpu.h"

class Material;
class Texture;
class Transform;
class Camera;

class Mesh
{
public:
	Mesh() = default;
	Mesh(std::vector<Vertex_In>& vertices, std::vector<unsigned int>& indices);
	~Mesh();

	Mesh(const Mesh& mesh);

	void SetMaterial(Material* pMaterial);
	void AddTexture(Texture* pTexture);
	void SetTransform(Transform* pTransform);
	void AddSubmesh(Mesh* pMesh);

	Material* GetMaterial() const;
	Transform* GetTransform() const;

	size_t GetSubmeshCount() const;
	Mesh* GetSubmesh(int id) const;

	void ReadyTextures();

	static void SetLinesRender(bool lines);
	static bool GetLinesRender();

	std::vector<Vertex_In>& GetVertices();
	std::vector<unsigned int>& GetIndices();
	std::vector<Texture*>& GetTextures();

	Vertex_In* GetVertexPointer();
	const unsigned int* GetIndexPointer() const;

	static Mesh* CreateCube(float width, float height, float depth);

private:
	std::vector<Vertex_In> m_Vertices;
	std::vector<unsigned int> m_Indices;
	std::vector<Mesh*> m_pSubmeshes;

	static bool m_Lines;

	unsigned int m_VBO;
	unsigned int m_VAO;
	unsigned int m_EBO;

	Material* m_pMaterial;
	Transform* m_pTransform;

	std::vector<Texture*> m_pTextures;
};

