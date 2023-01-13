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

void LightManager::GenerateLightmap()
{
    InitPointLights(m_PointLights);
    InitDirectionalLights(m_DirectionalLights);

    InitLightMap(m_LightmapSize);

    //GenerateLightMap();
}
