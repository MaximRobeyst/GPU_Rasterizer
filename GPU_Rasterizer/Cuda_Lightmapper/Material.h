#pragma once
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include <glm/glm.hpp>

class Texture;
class Material
{
public:
	Material(const char* vertexShaderSource, const char* fragmentShaderSource);
	~Material();

	unsigned int GetShaderProgram();

	void UseProgram();
	
	void SetBool(const std::string& name, bool value) const;
	void SetInt(const std::string& name, int value) const;
	void SetFloat(const std::string& name, float value) const;

	void SetTexture(const std::string& name, Texture* pTexture) const;
	void SetMat4(const std::string& name, const glm::mat4& matrix) const;
	void SetVec3(const std::string& name, const glm::vec3& vector) const;

private:
	void InitializeShaderProgram(const char* vertexShaderSource, const char* fragmentShaderSource);
	
	unsigned int m_ShaderProgram;

};

