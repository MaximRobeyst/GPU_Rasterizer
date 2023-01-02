#include "Texture.h"

#include <driver_types.h>
#include <texture_types.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>

Texture::Texture(const std::string& texturePath)
{
	m_pTexture = IMG_Load(texturePath.c_str()); 
	m_pTextureBuffer = static_cast<uint32_t*>(m_pTexture->pixels);
}

__host__
__device__
glm::vec3 Texture::Sample(const glm::vec2& uv)
{
	glm::ivec2 samplePoint{ };
	samplePoint.x = glm::clamp(static_cast<int>(uv.x * m_pTexture->w), 0, m_pTexture->w);
	samplePoint.y = glm::clamp(static_cast<int>(uv.y * m_pTexture->h), 0, m_pTexture->h);

	uint8_t r;
	uint8_t g;
	uint8_t b;

	uint32_t i = (samplePoint.x) + ((samplePoint.y) * m_pTexture->w);

	SDL_GetRGB(m_pTextureBuffer[i], m_pTexture->format, &r, &g, &b);

	return glm::vec3{ static_cast<float>(r) / 255.f,static_cast<float>(g) / 255.f,static_cast<float>(b) / 255.f };
}
