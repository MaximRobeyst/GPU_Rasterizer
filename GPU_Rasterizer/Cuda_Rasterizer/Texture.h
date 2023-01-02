#pragma once
#include <string>
#include <SDL_image.h>

#include <glm/glm.hpp>

#include <crt/host_defines.h>

class Texture
{
public:
	Texture(const std::string& name);

	__host__
	__device__
	glm::vec3 Sample(const glm::vec2& uv);

	__host__
	__device__
	inline	static glm::vec3 TextureSample(Texture* textures, glm::vec2 uv)
	{
		glm::ivec2 samplePoint{ };
		samplePoint.x = glm::clamp(static_cast<int>(uv.x * textures->m_pTexture->w), 0, textures->m_pTexture->w);
		samplePoint.y = glm::clamp(static_cast<int>(uv.y * textures->m_pTexture->h), 0, textures->m_pTexture->h);

		uint8_t r = 0;
		uint8_t g = 0;
		uint8_t b = 0;

		uint32_t i = (samplePoint.x) + ((samplePoint.y) * textures->m_pTexture->w);

		//SDL_GetRGB(textures->m_pTextureBuffer[i], textures->m_pTexture->format, &r, &g, &b);

		return glm::vec3{ static_cast<float>(r) / 255.f,static_cast<float>(g) / 255.f,static_cast<float>(b) / 255.f };
	}

	SDL_Surface* GetTexture() const { return m_pTexture; }
	uint32_t* GetBuffer() { return m_pTextureBuffer; }

private:
	SDL_Surface* m_pTexture;
	uint32_t* m_pTextureBuffer;
};

