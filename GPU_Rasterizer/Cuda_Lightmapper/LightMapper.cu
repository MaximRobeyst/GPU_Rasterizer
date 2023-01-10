#include "LightMapper.cuh"
#include <vector>

void InitMeshBuffers(Mesh* mesh)
{
	g_VertCount = mesh->GetVertices().size();
	g_IndicesCount = mesh->GetIndices().size();

	// Alocate vertex in buffer
	cudaFree(g_pVerteInBuffer);
	cudaMalloc(&g_pVerteInBuffer, g_VertCount * sizeof(Vertex));
	cudaMemcpy(g_pVerteInBuffer, mesh->GetVertices().data(), sizeof(Vertex) * g_VertCount, cudaMemcpyHostToDevice);

	// Allocate indices in buffer
	cudaFree(g_pIndexBuffer);
	cudaMalloc(&g_pIndexBuffer, g_IndicesCount * sizeof(unsigned int));
	cudaMemcpy(g_pIndexBuffer, mesh->GetIndices().data(), sizeof(unsigned int) * g_IndicesCount, cudaMemcpyHostToDevice);

	checkCUDAError("InitMeshBuffers");
}

void GenerateLightMap(const std::vector<Mesh*>& pMeshes)
{
	for (Mesh* pMesh : pMeshes)
	{
		// Init Mesh buffers
		InitMeshBuffers(pMesh);

		// Render Directional light
		


		// Render pointlight lightmaps


		// Assamble lightmaps
	}
}