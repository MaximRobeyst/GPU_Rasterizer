#include "Camera.h"
#include "const.h"
#include "gpu.h"
#include <glm/gtc/type_ptr.hpp>

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

	m_ProjectionMatrix = glm::mat4{
		glm::vec4{1 / (aspectRatio * FOV), 0,0,0},
		glm::vec4{0,1 / FOV, 0, 0},
		glm::vec4{0,0, -far / (far - near), -1},
		glm::vec4{0,0,-(far * near) / (far - near), 0}
	};
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
glm::mat4 Camera::GetProjectionMatrix()
{

	return m_ProjectionMatrix;
}

__host__
__device__
void Camera::UpdateMatrix()
{
	//FORWARD (zAxis) with YAW applied
	glm::mat3 yawRotation = MakeRotationY(m_AbsoluteRotation.y * float(E_TO_RADIANS));
	glm::vec3 zAxis = yawRotation * m_Forward;

	//Calculate RIGHT (xAxis) based on transformed FORWARD
	glm::vec3 xAxis = glm::normalize(glm::cross(glm::vec3{ 0.f,1.f,0.f }, zAxis));

	//FORWARD with PITCH applied (based on xAxis)
	glm::mat3 pitchRotation = MakeRotation(m_AbsoluteRotation.x * float(E_TO_RADIANS), xAxis);
	zAxis = pitchRotation * zAxis;

	//Calculate UP (yAxis)
	glm::vec3 yAxis = glm::cross(zAxis, xAxis);

	//Translate based on transformed axis
	m_Position += m_RelativeTranslation.x * xAxis;
	m_Position += m_RelativeTranslation.y * yAxis;
	m_Position += m_RelativeTranslation.z * zAxis;

	//Construct View2World Matrix
	m_ViewToWorld =
	{
		glm::vec4{xAxis, 0.0f},
		glm::vec4{yAxis, 0.0f},
		glm::vec4{zAxis, 0.0f},
		glm::vec4{m_Position.x,m_Position.y,m_Position.z,1.f}
	};

	//Construct World2View Matrix || viewMatrix
	m_WorldToView = glm::inverse(m_ViewToWorld);

	// WorldViewProjectionMatrix = ProjectionMatrix * ViewMatrix * WorldMatrix
	m_WorldViewProjectionMatrix = m_ProjectionMatrix * m_ViewToWorld;
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
