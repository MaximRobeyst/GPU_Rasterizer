#pragma once
#include <string>
#include <SDL_image.h>

#include <glm/glm.hpp>

#include <crt/host_defines.h>
#include <vector>

class Texture
{
public:
	Texture(const std::string& name);
	~Texture();

	unsigned char* GetData() const;
	size_t GetSize() const;

	int GetWidth() const;
	int GetHeight() const;

private:
	unsigned char* m_pImageData;
	int m_Width;
	int m_Height;
	int m_NrOfChannels;
};

