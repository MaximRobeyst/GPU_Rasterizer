#include "Texture.h"

#include <driver_types.h>
#include <texture_types.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

Texture::Texture(const std::string& texturePath)
{
	m_pImageData = stbi_load(texturePath.data(), &m_Width, &m_Height, &m_NrOfChannels, 0);
}

Texture::~Texture()
{
	stbi_image_free(m_pImageData);
}

unsigned char* Texture::GetData() const
{
	return m_pImageData;
}

size_t Texture::GetSize() const
{
	return m_Width * m_Height * m_NrOfChannels;
}

int Texture::GetWidth() const
{
	return m_Width;
}

int Texture::GetHeight() const
{
	return m_Height;
}
