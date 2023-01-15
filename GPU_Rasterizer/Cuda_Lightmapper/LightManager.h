#pragma once
#include <glm/glm.hpp>
#include <vector>
#include "LightRasterizer.h"

struct PointLight
{
	float radius;
	float strength;
	glm::vec3 position;
	glm::vec3 color;
};

struct DirectionalLight
{
	glm::vec3 direction;
	glm::vec3 color;
	float strength;
};

class LightManager
{
public:
	LightManager();

	static LightManager* GetLightManager();

	void AddPointLight(PointLight pointLight);
	void SetDirectionalLight(DirectionalLight directionalLight);

	int GetNumberOfPointLights() const;
	PointLight GetPointLight(int i) const;

	void GenerateLightmap(const std::vector<Mesh*>pMeshes);
private:
	LightRasterizer m_LightRasterizer{};

	static LightManager* m_pLightMapper;

	int m_LightmapSize = 128;

	std::vector<PointLight> m_PointLights;
	std::vector<DirectionalLight> m_DirectionalLights;
};

