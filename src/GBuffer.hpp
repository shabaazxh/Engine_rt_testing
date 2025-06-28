#pragma once
#include <volk/volk.h>
#include <memory>
#include <unordered_map>
#include "Camera.hpp"

namespace vk
{
	class Context;
	class Scene;
	class Buffer;

	class GBuffer
	{
	public:

		struct GBufferMRT
		{
			Image Albedo;
			Image Normal;
			Image WorldPositions;
			Image MetallicRoughness;
			Image Depth;
		};

		GBuffer(Context& context, std::shared_ptr<Scene>& scene, std::shared_ptr<Camera>& camera);
		~GBuffer();
		void Execute(VkCommandBuffer cmd);
		void Update();
		void Resize();

		GBufferMRT& GetGBufferMRT() { return m_GBufferMRT; }

	private:
		void CreatePipeline();
		void CreateRenderPass();
		void CreateFramebuffer();
		void BuildDescriptors();

		GBufferMRT m_GBufferMRT;

		VkRenderPass m_renderPass;
		VkFramebuffer m_framebuffer;
		VkDescriptorSetLayout m_descriptorSetLayout;

		Context& context;

		std::shared_ptr<Scene> scene;
		std::shared_ptr<Camera> camera;
		std::vector<VkDescriptorSet> m_descriptorSets;

		VkPipeline m_Pipeline;
		VkPipelineLayout m_PipelineLayout;

		VkPipeline m_AlphaMaskingPipeline;
		VkPipelineLayout m_AlphaMaskingPipelineLayout;
	};
}