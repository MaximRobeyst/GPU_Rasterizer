#include "Camera.h"
#include "Transform.h"

Camera::Camera(float fovAngle, float aspectRatio, float nearClipping, float farClipping, Transform* pTransform)
	: m_Projection{glm::perspective(glm::radians(fovAngle / 2.f),aspectRatio, nearClipping, farClipping)}
	, m_pTransform{pTransform}
	, m_Zoom{fovAngle / 2.f}
	, m_Aspect{aspectRatio}
{
}

Camera::~Camera()
{
	delete m_pTransform;
}

glm::mat4 Camera::GetViewMatrix()
{
	glm::mat4 view = glm::mat4{ 1.0f };
	glm::vec3 position = m_pTransform->GetPosition();

	view = glm::lookAt(m_pTransform->GetPosition(), m_pTransform->GetPosition() + m_Forward, m_Up);

	return view;
}

glm::mat4 Camera::GetProjectionMatrix() const
{
	return m_Projection;
}

glm::mat4 Camera::RegenerateProjectionMatrix(float fovAngle, float aspectRatio, float nearClipping, float farClipping)
{
	m_Zoom = fovAngle;
	m_Aspect = aspectRatio;
	m_Projection = glm::perspective(glm::radians(fovAngle / 2.f), aspectRatio, nearClipping, farClipping);
	return m_Projection;
}

Transform* Camera::GetTransform() const
{
	return m_pTransform;
}

float Camera::GetZoom() const
{
	return m_Zoom;
}

void Camera::ProcessKeyboard(char key, float deltaTime)
{
	const float cameraSpeed = 2.5f * deltaTime;
	glm::vec3 cameraPosition = GetTransform()->GetPosition();
	if (key == 'w')
		cameraPosition += cameraSpeed * m_Forward;
	if(key == 's')
		cameraPosition -= cameraSpeed * m_Forward;
	if (key == 'a')
		cameraPosition -= cameraSpeed * m_Right;
	if (key == 'd')
		cameraPosition += cameraSpeed * m_Right;

	GetTransform()->SetPosition(cameraPosition);
}

void Camera::ProcessMouseMovement(float xoffset, float yoffset, float deltaTime,GLboolean constrainPitch)
{
	const float sensitivity = 0.1f;
	xoffset *= sensitivity;
	yoffset *= sensitivity;

	m_Yaw += xoffset;
	m_Pitch += yoffset;

	if (constrainPitch)
	{
		if (m_Pitch > 89.0f)
			m_Pitch = 89.f;
		if (m_Pitch < -89.f)
			m_Pitch = -89.f;
	}

	UpdateCameraVectors();
}

void Camera::ProcessMouseScroll(float yOffset)
{
	m_Zoom -= yOffset;
	if (m_Zoom < 1.0f)
		m_Zoom = 1.0f;
	if (m_Zoom > 45.0f)
		m_Zoom = 45.0f;

	RegenerateProjectionMatrix(m_Zoom, m_Aspect);
}

void Camera::UpdateCameraVectors()
{
	// calculate the new Front vector
	glm::vec3 front;
	front.x = cos(glm::radians(m_Yaw)) * cos(glm::radians(m_Pitch));
	front.y = sin(glm::radians(m_Pitch));
	front.z = sin(glm::radians(m_Yaw)) * cos(glm::radians(m_Pitch));
	m_Forward = glm::normalize(front);
	// also re-calculate the Right and Up vector
	m_Right = glm::normalize(glm::cross(m_Forward, glm::vec3{0,1,0}));  // normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
	m_Up = glm::normalize(glm::cross(m_Right, m_Forward));
}
