#pragma once
#include <volk/volk.h>
#include <memory>
#include "Image.hpp"
#include <vector>
#include "GBuffer.hpp"

namespace vk
{
	class Context;
	class Camera;
	class Scene;
	class Spatial
	{
	public:
		explicit Spatial(Context& context, std::shared_ptr<Scene>& scene, std::shared_ptr<Camera>& camera, Image& initialCandidates, Image& TemporalReuseReservoirs, const GBuffer::GBufferMRT& gbufferMRT);
		~Spatial();

		void Execute(VkCommandBuffer cmd);
		void Update();
		void Resize();

		Image& GetRenderTarget() { return m_RenderTarget; }
		Image& GetSpatialReuseReservoirs() { return m_SpatialReuseReservoirs; }
	private:
		void CreatePipeline();
		void CreateShaderBindingTable();
		void BuildDescriptors();

		Context& context;
		std::shared_ptr<Scene> scene;
		std::shared_ptr<Camera> camera;
		const GBuffer::GBufferMRT& gbufferMRT;
		Image m_RenderTarget;
		Image m_SpatialReuseReservoirs;
		Image& initialCandidates;
		Image& TemporalReuseReservoirs;

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
	};
}