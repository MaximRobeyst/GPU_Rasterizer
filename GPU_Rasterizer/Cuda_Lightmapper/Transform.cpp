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

void Transform::SetRotation(float angle, glm::vec3 axis)
{
    m_Dirty = true;
}

glm::mat4 Transform::GetWorldTransform()
{
    if(!m_Dirty)
        return m_WorldTransform;

    m_WorldTransform = glm::translate(m_WorldTransform, m_Position);

    m_Dirty = false;
    return m_WorldTransform;
}

void Transform::SetWorldTransform(glm::mat4 matrix)
{
    m_WorldTransform = matrix;
}
