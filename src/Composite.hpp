#pragma once
#include <volk/volk.h>
#include <memory>
#include "Image.hpp"
#include <vector>

namespace vk
{
	class Context;

	class Composite
	{
	public:
		explicit Composite(Context& context, Image& LightingPass, Image& BloomPass, const std::vector<Image>& historyImages);
		~Composite();

		void Execute(VkCommandBuffer cmd);
		void Update();
		void Resize();

		Image& GetRenderTarget() { return m_RenderTarget; }

	private:
		void CreatePipeline();
		void CreateRenderPass();
		void CreateFramebuffer();
		void BuildDescriptors();

		Context& context;
		Image m_RenderTarget;
		Image& LightingPass;
		Image& BloomPass;
		const std::vector<Image>& historyImages;

		VkPipeline m_Pipeline;
		VkPipelineLayout m_PipelineLayout;
		std::vector<VkDescriptorSet> m_descriptorSets;
		VkDescriptorSetLayout m_descriptorSetLayout;
		VkRenderPass m_renderPass;
		VkFramebuffer m_framebuffer;

		uint32_t m_width;
		uint32_t m_height;
	};
}