#pragma once
#include <vector>
#include <glm/glm.hpp>

class Material;
class Texture;
class Transform;
class Camera;

struct Vertex
{
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec2 uv;
};

class Mesh
{
public:
	Mesh() = default;
	Mesh(std::vector<Vertex>& vertices, std::vector<unsigned int>& indices);
	~Mesh();

	void Initialize();
	void Draw(Camera* pCamera);

	void SetMaterial(Material* pMaterial);
	void AddTexture(Texture* pTexture);
	void SetTransform(Transform* pTransform);
	void AddSubmesh(Mesh* pMesh);

	Material* GetMaterial() const;

	size_t GetSubmeshCount() const;
	Mesh* GetSubmesh(int id) const;

	void ReadyTextures();

	static void SetLinesRender(bool lines);
	static bool GetLinesRender() ;

private:
	std::vector<Vertex> m_Vertices;
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

