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

	class SpatialCompute
	{
	public:
		explicit SpatialCompute(Context& context, std::shared_ptr<Scene>& scene, std::shared_ptr<Camera>& camera, Image& initial_candidates, Image& temporal_pass_reservoirs, const GBuffer::GBufferMRT& gbufferMRT);
		~SpatialCompute();

		void Execute(VkCommandBuffer cmd);
		void Update();
		void Resize();

		Image& GetRenderTarget() { return m_RenderTarget; }
		Image& GetShadingResult() { return m_TemporaryShadingResult; }
	private:
		void CreatePipeline();
		void BuildDescriptors();

		Context& context;
		std::shared_ptr<Scene> scene;
		std::shared_ptr<Camera> camera;
		Image m_RenderTarget;
		Image m_TemporaryShadingResult;

		Image& initial_candidates;
		Image& temporal_pass_reservoirs;
		const GBuffer::GBufferMRT& gbufferMRT;

		VkPipeline m_Pipeline;
		VkPipelineLayout m_PipelineLayout;
		std::vector<VkDescriptorSet> m_descriptorSets;
		VkDescriptorSetLayout m_descriptorSetLayout;

		uint32_t m_width;
		uint32_t m_height;

		std::vector<Buffer> m_uniformBuffers;
	};
}