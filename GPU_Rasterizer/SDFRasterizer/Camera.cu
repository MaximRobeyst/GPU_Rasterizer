#include "Camera.h"
#include "const.h"

#define GLM_FORCE_CUDA
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <iostream>
#include <SDL.h>

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
	m_ProjectionMatrix = glm::perspective(glm::radians(FOV), aspectRatio, near, far);

	UpdateMatrix();
}

Camera::~Camera()
{
}

__host__
__device__
glm::mat4 Camera::GetWorldMatrix()
{
	return m_WorldMatrix;
}

__host__
__device__
glm::mat4 Camera::GetViewMatrix()
{
	return m_ViewMatrix;
}

__host__
__device__
glm::mat4 Camera::GetProjectionMatrix()
{
	return m_ProjectionMatrix;
}

__host__
__device__
glm::mat4 Camera::GetWorldViewProjectionMatrix()
{
	return m_WorldViewProjectionMatrix * m_WorldMatrix;
}

void Camera::UpdatePosition(const glm::vec3& difference)
{
	m_Position += difference;

	UpdateMatrix();

	std::cout << "(" << m_Position.x << ", " << m_Position.y << ", " << m_Position.z << ")" << std::endl;
}

void Camera::Update(float elapsedSec)
{
	//Capture Input (absolute) Rotation & (relative) Movement
	//*************
	//Keyboard Input
	const uint8_t* pKeyboardState = SDL_GetKeyboardState(0);
	float keyboardSpeed = pKeyboardState[SDL_SCANCODE_LSHIFT] ? m_KeyboardMoveSensitivity * m_KeyboardMoveMultiplier : m_KeyboardMoveSensitivity;
	m_RelativeTranslation.x = (pKeyboardState[SDL_SCANCODE_D] - pKeyboardState[SDL_SCANCODE_A]) * keyboardSpeed * elapsedSec;
	m_RelativeTranslation.y = 0;
	m_RelativeTranslation.z = (pKeyboardState[SDL_SCANCODE_W] - pKeyboardState[SDL_SCANCODE_S]) * keyboardSpeed * elapsedSec;

	//Mouse Input
	int x, y = 0;
	uint32_t mouseState = SDL_GetRelativeMouseState(&x, &y);
	if (mouseState == SDL_BUTTON_LMASK)
	{
		m_RelativeTranslation.z += y * m_MouseMoveSensitivity * elapsedSec;
		m_AbsoluteRotation.y -= x * m_MouseRotationSensitivity;
	}
	else if (mouseState == SDL_BUTTON_RMASK)
	{
		m_AbsoluteRotation.x -= y * m_MouseRotationSensitivity;
		m_AbsoluteRotation.y -= x * m_MouseRotationSensitivity;
	}
	else if (mouseState == (SDL_BUTTON_LMASK | SDL_BUTTON_RMASK))
	{
		m_RelativeTranslation.y -= y * m_MouseMoveSensitivity * elapsedSec;
	}

	//Update LookAt (view2world & world2view matrices)
	//*************
	UpdateMatrix();
}

__host__
__device__
void Camera::UpdateMatrix()
{
	//FORWARD (zAxis) with YAW applied
	//glm::mat3 yawRotation = MakeRotationY(m_AbsoluteRotation.y * float(E_TO_RADIANS));
	glm::mat3 yawRotation = glm::eulerAngleY(glm::radians(m_AbsoluteRotation.y));

	glm::vec3 zAxis = yawRotation * m_Forward;

	//Calculate RIGHT (xAxis) based on transformed FORWARD
	glm::vec3 xAxis = glm::normalize(glm::cross(zAxis, glm::vec3{ 0.f,1.f,0.f }));

	//FORWARD with PITCH applied (based on xAxis)

	glm::mat4 identity = glm::mat4(1.0f);
	glm::mat3 pitchRotation = glm::rotate(identity,glm::radians(m_AbsoluteRotation.x), xAxis);
	zAxis = pitchRotation * zAxis;

	//Calculate UP (yAxis)
	glm::vec3 yAxis = glm::cross(xAxis, zAxis);

	//Translate based on transformed axis
	m_Position += m_RelativeTranslation.x * xAxis;
	m_Position += m_RelativeTranslation.y * yAxis;
	m_Position += m_RelativeTranslation.z * zAxis;

	//Construct View2World Matrix
	m_ViewMatrix = glm::lookAt(m_Position, m_Position + zAxis, yAxis);
	//Construct World2View Matrix || viewMatrix
	m_WorldMatrix = glm::inverse(m_ViewMatrix);

	// WorldViewProjectionMatrix = ProjectionMatrix * ViewMatrix * WorldMatrix
	m_WorldViewProjectionMatrix = m_ProjectionMatrix * m_ViewMatrix;
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
