#pragma once
#include <glm\glm.hpp>
#include <cuda_runtime.h>

class Camera final
{
public:
	__device__ Camera(const glm::vec3& position, const glm::vec3& forward, float FOV, float aspectRatio, float far = 100.0f, float near = 0.1f);

	__device__ glm::mat4 GetWorldMatrix();
	__device__ glm::mat4 GetProjectionMatrix();

private:
	__device__ void UpdateMatrix();
	__device__ glm::mat3 MakeRotationY(float f);
	__device__ glm::mat3 MakeRotation(float f, glm::vec3 axis);

	static Camera* m_MainCamera;

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