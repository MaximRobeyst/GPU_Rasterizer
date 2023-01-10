#pragma once
#include <glm/glm.hpp>
#include <vector>

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

	void GenerateLightmap();
private:
	static LightManager* m_pLightMapper;

	int m_LightmapSize = 128;

	std::vector<PointLight> m_PointLights;
	std::vector<DirectionalLight> m_DirectionalLights;
};

