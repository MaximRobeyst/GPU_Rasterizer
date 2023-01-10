#include "Transform.h"


Transform::Transform(glm::vec3 position)
    : m_Position{position}
{

}

glm::vec3 Transform::GetPosition() const
{
    return m_Position;
}

void Transform::SetPosition(glm::vec3 position)
{
    m_Position = position;
    m_Dirty = true;
}

void Transform::SetRotation(float x, float y, float z)
{
    m_Dirty = true;
    m_Rotation.x = x;
    m_Rotation.y = y;
    m_Rotation.z = z;
}

void Transform::SetRotation(float angle, glm::vec3 axis)
{
    m_Dirty = true;
}

glm::mat4 Transform::GetWorldTransform()
{
    if(!m_Dirty)
        return m_WorldTransform;

    m_WorldTransform = glm::mat4{ 1.0f };
    m_WorldTransform = glm::translate(m_WorldTransform, m_Position);
    m_WorldTransform = glm::rotate(m_WorldTransform, m_Rotation.y, glm::vec3{ 0,1,0 });

    m_Dirty = false;
    return m_WorldTransform;
}

void Transform::SetWorldTransform(glm::mat4 matrix)
{
    m_WorldTransform = matrix;
}
