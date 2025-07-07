#pragma once

#include <volk/volk.h>
#include "Image.hpp"
#include <vector>


namespace vk
{
	class Context;
	class Camera;
	class Buffer;
	struct CameraTransform;

	class MotionVectors
	{
	public:
		MotionVectors(Context& context, std::shared_ptr<Camera> camera, Image& GBufferWorldPosition);
		~MotionVectors();
		void Update();
		void Resize();
		void Execute(VkCommandBuffer cmd);

		Image& GetRenderTarget() { return m_RenderTarget; }

	private:
		void CreatePipeline();
		void CreateRenderPass();
		void CreateFramebuffer();
		void CreateDescriptors();
	private:
		Context& context;
		std::shared_ptr<Camera> camera;
		Image& GBufferWorldPosition;

		uint32_t m_width;
		uint32_t m_height;
		Image m_RenderTarget;

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