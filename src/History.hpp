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
		void Execute(VkCommandBuffer cmd);


		std::vector<Image>& GetHistoryImages() { return m_historyImages; }
		Image& GetRenderTarget() { return m_RenderTarget; }

		void Destroy();
		void Update();
	private:
		void CreatePipeline();
		void CreateRenderPass();
		void BuildDescriptors();
		void CreateFramebuffers();

	private:
		Context& context;
		const Image& renderedImage;

		uint32_t m_FrameToWriteTo;
		uint32_t m_width;
		uint32_t m_height;

		std::vector<Image> m_historyImages;
		VkFramebuffer m_historyFramebuffers[5];
		VkRenderPass m_renderPass;

		Image m_RenderTarget;

		VkPipeline m_pipeline;
		VkPipelineLayout m_pipelineLayout;
		VkDescriptorSetLayout m_descriptorSetLayout;
		std::vector<VkDescriptorSet> m_descriptorSets;
		std::vector<Buffer> m_rtxSettingsUBO;
		uint32_t accNumber;
	};
}