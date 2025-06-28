#pragma once
#include <volk/volk.h>
#include <memory>
#include "Image.hpp"
#include <vector>

namespace vk
{
	class Context;
	class Camera;
	class Scene;
	class RayPass
	{
	public:
		explicit RayPass(Context& context, std::shared_ptr<Scene>& scene, std::shared_ptr<Camera>& camera);
		~RayPass();

		void Execute(VkCommandBuffer cmd);
		void Update();
		void Resize();

		Image& GetRenderTarget() { return m_RenderTarget; }
		Image& GetInitialCandidates() { return m_InitialCandidates; }
		Image& GetWorldHitPositions() { return m_WorldPositionsTarget; }
		Image& GetHitNormals() { return m_NormalsTarget; }
		Image& GetAlbedo() { return m_AlbedoTarget; }

	private:
		void CreatePipeline();
		void CreateShaderBindingTable();
		void BuildDescriptors();

		Context& context;
		std::shared_ptr<Scene> scene;
		std::shared_ptr<Camera> camera;
		Image m_RenderTarget;
		Image m_WorldPositionsTarget;
		Image m_NormalsTarget;
		Image m_AlbedoTarget;
		Image m_InitialCandidates;

		VkPipeline m_Pipeline;
		VkPipelineLayout m_PipelineLayout;
		std::vector<VkDescriptorSet> m_descriptorSets;
		VkDescriptorSetLayout m_descriptorSetLayout;

		uint32_t m_width;
		uint32_t m_height;
		VkPhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingPipelineProperties{};
		std::unique_ptr<Buffer> RayGenShaderBindingTable;
		std::unique_ptr<Buffer> MissShaderBindingTable;
		std::unique_ptr<Buffer> HitShaderBindingTable;

		std::vector<Buffer> m_rtxSettingsUBO;
		Buffer m_Reservoirs;
	};
}