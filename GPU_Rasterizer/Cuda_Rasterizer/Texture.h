#pragma once
#include <string>
#include <SDL_image.h>

#pragma warning(push, 0)
#include <glm\glm.hpp>
#pragma warning(pop)

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
	int GetChannels() const;

private:
	std::string m_Path{};

	unsigned char* m_pImageData;
	int m_Width;
	int m_Height;
	int m_NrOfChannels;
};

