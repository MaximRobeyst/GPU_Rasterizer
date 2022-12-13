#include "Camera.h"

Camera::Camera(const glm::vec3& position, const glm::vec3& forward, float FOV, float aspectRatio, float far, float near)
{

	m_ProjectionMatrix = glm::mat4{
		glm::vec4{1 / (aspectRatio * FOV), 0,0,0},
		glm::vec4{0,1 / FOV, 0, 0},
		glm::vec4{0,0, -far / (far - near), -1},
		glm::vec4{0,0,-(far * near) / (far - near), 0}
	};
}

void Camera::UpdateMatrix()
{
}
