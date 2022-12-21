#include "Camera.h"
#include "const.h"

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

Camera::Camera()
	: m_AspectRatio{SCREEN_WIDTH / SCREEN_HEIGHT}
	, m_Far{1000.0f}
	, m_Near{0.1f}
	, m_FOV{60 * static_cast<float>(E_TO_RADIANS) }
{
}

__host__
__device__
Camera::Camera(const glm::vec3& position, const glm::vec3& forward, float FOV, float aspectRatio, float far, float near)
	: m_Position{position}
	, m_Forward{forward}
	, m_FOV{FOV * static_cast<float>(E_TO_RADIANS)}
	, m_AspectRatio{aspectRatio}
	, m_Far{far}
	, m_Near{near}
{

	m_WorldToView = glm::mat4(1.0f);
	m_ProjectionMatrix = glm::perspective(glm::radians(60.0f), aspectRatio, near, far);

	UpdateMatrix();
}

Camera::~Camera()
{
}

__host__
__device__
glm::mat4 Camera::GetWorldMatrix()
{
	return m_WorldToView;
}

__host__
__device__
glm::mat4 Camera::GetWorldViewProjectionMatrix()
{

	return m_WorldViewProjectionMatrix * m_WorldToView;
}

__host__
__device__
void Camera::UpdatePosition(const glm::vec2& difference)
{
	m_Position.x += difference.x;
	m_Position.z += difference.y;

	UpdateMatrix();
}

__host__
__device__
void Camera::UpdateMatrix()
{
	// WorldViewProjectionMatrix = ProjectionMatrix * ViewMatrix * WorldMatrix
	m_WorldViewProjectionMatrix = m_ProjectionMatrix * m_ViewMatri;
}

__host__
__device__
glm::mat3 Camera::MakeRotationY(float t)
{
	float c = cosf(t);
	float s = sinf(t);

	return glm::mat3(
		c, 0, s,
		0,1,0,
		-s, 0, c);
}

__host__
__device__
glm::mat3 Camera::MakeRotation(float t, glm::vec3 axis)
{
	float c = cosf(t);
	float s = sinf(t);
	float oneMinusC = (1 - c);

	float xy = axis.x * axis.y;
	float yz = axis.y * axis.z;
	float xz = axis.x * axis.z;
	float sx = s * axis.x;
	float sy = s * axis.y;
	float sz = s * axis.z;

	return glm::mat3(
		c + oneMinusC * (axis.x * axis.x), oneMinusC * xy - sz, oneMinusC * xz + sy,
		oneMinusC * xy + sz, c + oneMinusC * (axis.y * axis.y), oneMinusC * yz - sx,
		oneMinusC * xz - sy, oneMinusC * yz + sx, c + oneMinusC * (axis.z * axis.z));
}
