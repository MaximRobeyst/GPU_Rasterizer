#pragma once
class Texture
{
public:
	Texture(const char* texturePath, bool png = false);
	~Texture() = default;

	unsigned int GetTextureID();

private:
	unsigned int m_Texture;
};

