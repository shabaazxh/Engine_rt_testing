#pragma once
#include <volk/volk.h>
#include <memory>
#include "Camera.hpp"

namespace vk
{
	class Context;
	class Scene;
	class Buffer;

	class PresentPass
	{
	public:

		PresentPass(Context& context, Image& raypass_result, Image& composited_result);
		~PresentPass();
		void Execute(VkCommandBuffer cmd, uint32_t imageIndex);
		void Update();
		void Resize();
	private:
		void CreatePipeline();
		void BuildDescriptors();

		Context& context;
		Image& raypass_result;
		Image& composited_result;

		std::shared_ptr<Scene> scene;
		std::shared_ptr<Camera> camera;

		VkPipeline m_pipeline;
		VkPipelineLayout m_pipelineLayout;
		std::vector<VkDescriptorSet> m_descriptorSets;
		VkDescriptorSetLayout m_descriptorSetLayout;
		std::vector<Buffer> m_postProcessUbo;
		RenderType m_renderType;
	};
}