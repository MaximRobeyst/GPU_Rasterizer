#pragma once
#include <glm\glm.hpp>

class Camera final
{
public:
	Camera(const glm::vec3& position, const glm::vec3& forward, float FOV, float aspectRatio, float far = 100.0f, float near = 0.1f);

private:
	void UpdateMatrix();
	glm::mat3 MakeRotationY(float f);
	glm::mat3 MakeRotation(float f, glm::vec3 axis);

	const float m_AspectRatio;
	const float m_Far;
	const float m_Near;

	float m_FOV;

	glm::vec3 m_Position;
	glm::vec3 m_Forward{};

	glm::vec2 m_AbsoluteRotation{}; //Pitch(x) & Yaw(y) only
	glm::vec3 m_RelativeTranslation{};

	glm::mat4 m_WorldToView{};
	glm::mat4 m_ViewToWorld{};
	glm::mat4 m_ProjectionMatrix{};
	glm::mat4 m_WorldViewProjectionMatrix{};
};