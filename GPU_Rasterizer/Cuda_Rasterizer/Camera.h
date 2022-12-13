#pragma once
#include <glm\glm.hpp>

class Camera final
{
public:
	Camera(const glm::vec3& position, const glm::vec3& forward, float FOV, float aspectRatio, float far = 100.0f, float near = 0.1f);

private:
	void UpdateMatrix();

	glm::mat4 m_WorldToView{};
	glm::mat4 m_ViewToWorld{};
	glm::mat4 m_ProjectionMatrix{};
	glm::mat4 m_WorldViewProjectionMatrix{};
};