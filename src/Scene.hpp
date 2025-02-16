#pragma once

#include "Context.hpp"
#include "Image.hpp"
#include "Utils.hpp"
#include "Light.hpp"
#include "Buffer.hpp"

#include <cstddef>
#include <memory>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "GLTF.hpp"
#include <memory>

namespace vk
{
	struct AccelerationStructure
	{
		VkAccelerationStructureKHR handle;
		uint64_t deviceAddress;
		std::unique_ptr<Buffer> buffer;
	};

	class Scene
	{
	public:

		Scene(Context& context, MaterialManager& materialManager);
		void AddModel(GLTFModel& GLTF, MaterialManager& materialManager);

		// TODO: Sort and implement these 
		void RenderFrontMeshes(VkCommandBuffer cmd, VkPipelineLayout pipelineLayout);
		void RenderBackMeshes(VkCommandBuffer cmd, VkPipelineLayout pipelineLayout);

		void DrawGLTF(VkCommandBuffer cmd, VkPipelineLayout pipelineLayout); // Does it make sense for this to take VkPipeline? 
		void AddLightSource(Light& LightSource);
		void Update(GLFWwindow* window);

		void Destroy();

		std::vector<Light>&							   GetLights() { return m_Lights; }
		std::vector<Buffer>&						   GetLightsUBO() { return m_LightUBO; }

		void CreateBLAS();
		void CreateTLAS();
		AccelerationStructure BottomLevelAccelerationStructure;
		AccelerationStructure TopLevelAccelerationStructure;
		std::vector<GLTFModel> gltfModels;



		Buffer vertexBuffer;
		Buffer indexBuffer;
		Buffer meshOffsetBuffer;
	private:
		Context& context;
		MaterialManager& materialManager;

		std::vector<size_t> m_FrontMeshes;
		std::vector<size_t> m_BackMeshes;
		std::vector<Light>  m_Lights;
		LightBuffer m_LightBuffer;
		std::vector<Buffer> m_LightUBO;

	};
}