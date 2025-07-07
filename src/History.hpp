#pragma once

#include <volk/volk.h>
#include "Image.hpp"
#include <vector>

namespace vk
{
	class Context;
	class History
	{
	public:
		History(Context& context, const Image& renderedImage);
		~History();
		void Execute(VkCommandBuffer cmd);
		Image& GetRenderTarget() { return m_RenderTarget; }

		void Update();
		void Resize();
	private:
		void CreatePipeline();
		void BuildDescriptors();

	private:
		Context& context;
		const Image& renderedImage;

		uint32_t m_FrameToWriteTo;
		uint32_t m_width;
		uint32_t m_height;

		VkRenderPass m_renderPass;

		Image m_RenderTarget;

		VkPipeline m_pipeline;
		VkPipelineLayout m_pipelineLayout;
		VkDescriptorSetLayout m_descriptorSetLayout;
		std::vector<VkDescriptorSet> m_descriptorSets;
		std::vector<Buffer> m_rtxSettingsUBO;
	};
}