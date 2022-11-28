#include "Camera.h"
#include "const.h"

Camera::Camera(float3 position, float3 forward, float FOV, float far, float near)
    : m_World{1,0,0,0,
              0,1,0,0,
              0,0,1,0,
              0,0,0,1}
{
    m_ViewProjection = float4x4{
        1.0f / (static_cast<float>(SCREEN_WIDTH) / static_cast<float>(SCREEN_HEIGHT)), 0, 0, 0,
        0, 1.0f / FOV, 0,0,
        0, 0, -far / (far - near), -1,
        0, 0, -(far * near) / (far - near), 0
    };
}

__location__(device)float4x4 Camera::GetWorldViewProjectionMatri()
{
    return m_World * m_ViewProjection;
}

__device__ float4x4 Camera::GetViewProjectionMatri()
{
    return m_ViewProjection;
}

__location__(device)float4x4 Camera::GetWorldMatri()
{
    return m_World;
}
