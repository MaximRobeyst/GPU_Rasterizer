#pragma once
#include "Mesh.h"
#include <cuda.h>

void MakeSDF(Mesh* pMesh, int size, float* Texture3D);

class SDFMesh
{
public:
	SDFMesh(Mesh* pMesh, int size);	// Create SDF from 
	~SDFMesh();



private:
	void GenerateSDF(Mesh* pMesh, int size);

	Mesh* m_pMesh;

	unsigned int m_TextureSize{ 128 };
	std::vector<float> m_p3DTexture;
};

