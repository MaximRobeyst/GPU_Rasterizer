#include "SDFMesh.h"
#include <algorithm>

SDFMesh::SDFMesh(Mesh* pMesh, int size)
	: m_pMesh{pMesh}
{
    m_p3DTexture.resize(size * size * size);
    GenerateSDF(pMesh, size);
}

SDFMesh::~SDFMesh()
{
}


__host__
__device__
float dot2(glm::vec3 v) { return dot(v, v); }

__host__
__device__
float clamp(float value, float min, float max) { if (value < min) return min; if (value > max) return max; return value; }

__host__
__device__
float udTriangle(glm::vec3 v1, glm::vec3 v2, glm::vec3 v3, glm::vec3 p)
{
    // prepare data    
    glm::vec3 v21 = v2 - v1; glm::vec3 p1 = p - v1;
    glm::vec3 v32 = v3 - v2; glm::vec3 p2 = p - v2;
    glm::vec3 v13 = v1 - v3; glm::vec3 p3 = p - v3;
    glm::vec3 nor = cross(v21, v13);

    return sqrt( // inside/outside test    
        (glm::sign(dot(cross(v21, nor), p1)) +
            glm::sign(dot(cross(v32, nor), p2)) +
            glm::sign(dot(cross(v13, nor), p3)) < 2.0)
        ?
        // 3 edges    
        min(min(
            dot2(v21 * clamp(dot(v21, p1) / dot2(v21), 0.0, 1.0) - p1),
            dot2(v32 * clamp(dot(v32, p2) / dot2(v32), 0.0, 1.0) - p2)),
            dot2(v13 * clamp(dot(v13, p3) / dot2(v13), 0.0, 1.0) - p3))
        :
        // 1 face    
        dot(nor, p1) * dot(nor, p1) / dot2(nor));
}

void SDFMesh::GenerateSDF(Mesh* pMesh, int size)
{
    auto verts = pMesh->GetVertices();
    auto indices = pMesh->GetIndices();

    float fmin{ FLT_MAX };
    float fmax{ FLT_MIN };

    std::for_each(verts.begin(), verts.end(), [&](Vertex_In vert)
        {
            fmin = std::min(fmin, std::min(vert.position.x, std::min(vert.position.y, vert.position.z)));
            fmax = std::max(fmax, std::max(vert.position.x, std::max(vert.position.y, vert.position.z)));
        });

    glm::vec3 max{ fmax };
    glm::vec3 min{ fmin };

    for (int z = 0; z < size; ++z)
    {
        for (int y = 0; y < size; ++y)
        {
            for (int x = 0; x < size; ++x)
            {
                glm::vec3 position{};
                glm::vec3 delta = max - min;
                position.x = (delta.x / static_cast<float>(size)) * x;
                position.y = (delta.y / static_cast<float>(size)) * y;
                position.z = (delta.z / static_cast<float>(size)) * z;

                float smallestDistance = FLT_MAX;
                for (int i = 0; i < indices.size(); i += 3)
                {
                    auto v1 = verts[indices[i]].position;
                    auto v2 = verts[indices[i + 1]].position;
                    auto v3 = verts[indices[i + 2]].position;

                    float distance = udTriangle(v1, v2, v3, position);
                    if (distance < smallestDistance)
                        smallestDistance = distance;
                }

                m_p3DTexture[(x * size + y) * size + z] = smallestDistance;
            }
        }
    }
}

__global__
void GenerateSDF(int size, glm::vec3 min, glm::vec3 max, float* texture, Vertex_In* pVertices, unsigned int* pIndices, int indexCount)
{
    const int xPix = blockDim.x * blockIdx.x + threadIdx.x;
    const int yPix = blockDim.y * blockIdx.y + threadIdx.y;
    const int zPix = blockDim.z * blockIdx.z + threadIdx.z;

    if (xPix < size && yPix < size && zPix < size)
    {
        glm::vec3 position{};
        glm::vec3 delta = max - min;
        position.x = (delta.x / static_cast<float>(size)) * xPix;
        position.y = (delta.y / static_cast<float>(size)) * yPix;
        position.z = (delta.z / static_cast<float>(size)) * zPix;

        float smallestDistance = FLT_MAX;
        for (int i = 0; i < indexCount; i += 3)
        {
            auto v1 = pVertices[pIndices[i]].position;
            auto v2 = pVertices[pIndices[i + 1]].position;
            auto v3 = pVertices[pIndices[i + 2]].position;

            float distance = udTriangle(v1, v2, v3, position);
            if (distance < smallestDistance)
                smallestDistance = distance;
        }

        texture[(xPix * size + yPix) * size + zPix] = smallestDistance;
    }
}

void MakeSDF(Mesh* pMesh, int size, float* pTexture3D)
{
    auto verts = pMesh->GetVertices();
    auto indices = pMesh->GetIndices();

    float min{FLT_MIN};
    float max{FLT_MAX};

    std::for_each(verts.begin(), verts.end(), [&](Vertex_In vert)
        {
            min= std::min(min, std::min(vert.position.x, std::min(vert.position.y, vert.position.z)));
            max = std::max(max, std::max(vert.position.x, std::max(vert.position.y, vert.position.z)));
        });

    GenerateSDF << <256, 1, 1 >> > (size, glm::vec3{ min }, glm::vec3{ max }, pTexture3D, verts.data(), indices.data(), indices.size());
}
