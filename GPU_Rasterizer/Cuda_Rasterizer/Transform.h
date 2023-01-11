#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
class Transform
{
public:
	Transform(glm::vec3 position = glm::vec3{}, glm::vec3 scale = glm::vec3{1.0f});
	~Transform() = default;

	glm::vec3 GetPosition() const;

	// Position
	void SetPosition(glm::vec3 position);

	// Rotation
	void SetRotation(float x, float y, float z);
	void SetRotation(float angle, glm::vec3 axis);

	// Scale
	void SetScale(const glm::vec3& scale);

	glm::mat4 GetWorldTransform();
	void SetWorldTransform(glm::mat4 matrix);

private:
	bool m_Dirty{ true };

	glm::vec3 m_Position;
	glm::vec3 m_Rotation;
	glm::vec3 m_Scale;

	glm::mat4 m_WorldTransform {glm::mat4(1.0f)};

};

