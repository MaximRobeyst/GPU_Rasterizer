#pragma once
#include <glm/glm.hpp>
#include <GLFW/glfw3.h>

class Transform;
class Camera
{
public:
	Camera(float fovAngle, float aspectRatio, float nearClipping, float farClipping, Transform* pTransform);
	~Camera();

	glm::mat4 GetViewMatrix();
	glm::mat4 GetProjectionMatrix() const;

	glm::mat4 RegenerateProjectionMatrix(float fovAngle, float acpectRatio, float near = 0.1f, float farClipping = 100.0f);

	Transform* GetTransform() const;

	float GetZoom() const;

	void ProcessKeyboard(char key, float deltaTime);
	void ProcessMouseMovement(float xoffset, float yoffset, float deltaTime,GLboolean constrainPitch = true);
	void ProcessMouseScroll(float yOffset);

	static Camera* GetMainCamera();
private:
	void UpdateCameraVectors();

	static Camera* m_pMainCamera;

	glm::mat4 m_Projection;
	Transform* m_pTransform;

	glm::vec3 m_Forward{0,0,-1};
	glm::vec3 m_Up{0,1,0};
	glm::vec3 m_Right{1,0,0};

	float m_Yaw{};
	float m_Pitch{};

	float m_Zoom = 45.0f;
	float m_Aspect;
};

