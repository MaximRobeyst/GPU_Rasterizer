#include "LightManager.h"

#include "LightMapper.cuh"

LightManager* LightManager::m_pLightMapper{};

LightManager::LightManager()
{
}

LightManager* LightManager::GetLightManager()
{
    if (m_pLightMapper == nullptr) m_pLightMapper = new LightManager();

    return m_pLightMapper;
}

void LightManager::AddPointLight(PointLight pointLight)
{
    m_PointLights.emplace_back(pointLight);
}

void LightManager::SetDirectionalLight(DirectionalLight directionalLight)
{
    m_DirectionalLights.emplace_back(directionalLight);
}

int LightManager::GetNumberOfPointLights() const
{
    return m_PointLights.size();
}

PointLight LightManager::GetPointLight(int i) const
{
    return m_PointLights[i];
}

void LightManager::GenerateLightmap(const std::vector<Mesh*> pMeshes)
{
	const auto indices = pMeshes[0]->GetIndices();
	const auto vertices = pMeshes[0]->GetVertices();

	for (int i = 0; i < indices.size(); i += 3)
	{
		m_LightRasterizer.MapVerticesToCoords(vertices[indices[i]], vertices[indices[i + 1]], vertices[indices[i + 2]]);
	}
}
