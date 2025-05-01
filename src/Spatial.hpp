#pragma once

#include <volk/volk.h>
#include "Image.hpp"
#include <vector>

namespace vk
{
	class Context;
	class Spatial
	{
	public:
		Spatial(Context& context, const Image& initialCandidates);
		~Spatial();

		void Execute(VkCommandBuffer cmd);
		void Resize();

		void CreatePipeline();
		void CreateDescriptors();

	private:
		Context& context;
		const Image& initialCandidates;
		Image m_RenderTarget; // Will be the target with spatial re-use samples

		uint32_t m_width;
		uint32_t m_height;

		VkPipeline m_Pipeline;
		VkPipelineLayout m_PipelineLayout;
		VkDescriptorSetLayout m_DescriptorSetLayout;
		std::vector<VkDescriptorSet> m_Descriptors;
	};
};