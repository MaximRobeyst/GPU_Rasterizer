#pragma once
#include <cuda.h>

#define GLM_FORCE_CUDA
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm\glm.hpp>

#include <cuda_runtime.h>

class Camera final
{
public:
	__host__ __device__ Camera();
	__host__ __device__ Camera(const glm::vec3& position, const glm::vec3& forward, float FOV, float aspectRatio, float far = 100.0f, float near = 0.1f);
	__host__ __device__ ~Camera();

	__host__ __device__ glm::mat4 GetWorldMatrix();
	__host__ __device__ glm::mat4 GetViewMatrix();
	__host__ __device__ glm::mat4 GetProjectionMatrix();
	__host__ __device__ glm::mat4 GetWorldViewProjectionMatrix();

	void UpdatePosition(const glm::vec3& difference);

	float GetFar() const { return m_Far; }
	float GetNear() const { return m_Near; }

private:
	__host__ __device__ void UpdateMatrix();
	__host__ __device__ glm::mat3 MakeRotationY(float f);
	__host__ __device__ glm::mat3 MakeRotation(float f, glm::vec3 axis);

	const float m_AspectRatio;
	const float m_Far;
	const float m_Near;

	float m_FOV;

	glm::vec3 m_Position;
	glm::vec3 m_Forward{};

	glm::vec2 m_AbsoluteRotation{}; //Pitch(x) & Yaw(y) only
	glm::vec3 m_RelativeTranslation{};

	glm::mat4 m_WorldMatrix{};
	glm::mat4 m_ViewMatrix{};
	glm::mat4 m_ProjectionMatrix{};
	glm::mat4 m_WorldViewProjectionMatrix{};
};