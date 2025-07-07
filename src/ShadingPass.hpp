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

	class ShadingPass
	{
	public:
		explicit ShadingPass(
			Context& context,
			std::shared_ptr<Scene>& scene,
			std::shared_ptr<Camera>& camera,
			const GBuffer::GBufferMRT& gbufferMRT,
			Image& InitialCandidatesReservoirs,
			Image& TemporalPassReservoirs,
			Image& SpatialPassReservoirs
			);
		~ShadingPass();

		void Execute(VkCommandBuffer cmd);
		void Update();
		void Resize();

		Image& GetRenderTarget() { return m_RenderTarget; }

	private:
		void CreatePipeline();
		void BuildDescriptors();

		Context& context;
		std::shared_ptr<Scene> scene;
		std::shared_ptr<Camera> camera;
		const GBuffer::GBufferMRT& gbufferMRT;
		Image& InitialCandidatesReservoirs;
		Image& TemporalPassReservoirs;
		Image& SpatialPassReservoirs;

		Image m_RenderTarget;
		Image m_TemporaryShadingResult;

		VkPipeline m_Pipeline;
		VkPipelineLayout m_PipelineLayout;
		std::vector<VkDescriptorSet> m_descriptorSets;
		VkDescriptorSetLayout m_descriptorSetLayout;

		uint32_t m_width;
		uint32_t m_height;

		std::vector<Buffer> m_uniformBuffers;
	};
}