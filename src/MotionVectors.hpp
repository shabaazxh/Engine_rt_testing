#pragma once

#include <volk/volk.h>
#include "Image.hpp"
#include "Camera.hpp"

namespace vk
{
	class Context;
	struct CameraTransform;

	class MotionVectors
	{
	public:
		MotionVectors(Context& context, Camera& camera, const Image& currentDepthBuffer);
		~MotionVectors();
		void Update();
		void Resize();
		void Execute(VkCommandBuffer cmd);

	private:
		void CreatePipeline();
		void CreateRenderPass();
		void CreateFramebuffer();
		void CreateDescriptors();
	private:
		Context& context;
		Camera& camera;
		const Image& currentDepthBuffer;
		uint32_t m_width;
		uint32_t m_height;
		Image m_RenderTarget;
		CameraTransform m_previousCameraTransform;

		VkPipeline m_Pipeline;
		VkPipelineLayout m_PipelineLayout;
		VkFramebuffer m_Framebuffer;
		VkRenderPass m_RenderPass;
		VkDescriptorSetLayout m_DescriptorSetLayout;
		std::vector<VkDescriptorSet> m_DescriptorSets;
		std::vector<Buffer> m_previousCameraUB;
		CameraTransform m_PreviousCameraTransform;
	};
};