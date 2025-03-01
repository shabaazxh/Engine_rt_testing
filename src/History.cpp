#include "Context.hpp"
#include "Pipeline.hpp"
#include "RenderPass.hpp"
#include "History.hpp"

vk::History::History(Context& context, const Image& renderedImage) : context{context}, renderedImage{renderedImage}
{
	m_FrameToWriteTo = currentFrame;
	m_historyImages.resize(5);
	m_width = context.extent.width;
	m_height = context.extent.height;

	// Create the history buffer images
	for (size_t i = 0; i < 5; i++)
	{
		m_historyImages[i] = CreateImageTexture2D(
			"history_image " + std::to_string(i),
			context,
			m_width, m_height,
			VK_FORMAT_R32G32B32A32_SFLOAT,
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
			VK_IMAGE_ASPECT_COLOR_BIT
		);
	}

	for (size_t i = 0; i < 5; i++)
	{
		ExecuteSingleTimeCommands(context, [&](VkCommandBuffer cmd)
		{
			// Transition to TRANSFER_DST_OPTIMAL
			ImageTransition(
				cmd,
				m_historyImages[i].image,
				VK_FORMAT_R32G32B32A32_SFLOAT,
				VK_IMAGE_LAYOUT_UNDEFINED,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				0,
				VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
				VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
				VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
			);

			VkClearColorValue clearColor = { 1.0f, 0.0f, 0.0f, 1.0f }; // clear with red

			VkImageSubresourceRange imageRange = {};
			imageRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			imageRange.baseMipLevel = 0;
			imageRange.levelCount = 1;
			imageRange.baseArrayLayer = 0;
			imageRange.layerCount = 1;

			// clear the image before transitioning
			vkCmdClearColorImage(cmd, m_historyImages[i].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearColor, 1, &imageRange);

			// Transition the image
			ImageTransition(
				cmd,
				m_historyImages[i].image,
				VK_FORMAT_R32G32B32A32_SFLOAT,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				VK_ACCESS_TRANSFER_WRITE_BIT,
				VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT, // make sure it completes transfer
				VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
			);
		});
	}




	CreateRenderPass();
	CreateFramebuffers();
	BuildDescriptors();
	CreatePipeline();
}


void vk::History::Execute(VkCommandBuffer cmd)
{

#ifdef _DEBUG
	RenderPassLabel(cmd, "HistoryPass");
#endif // !DEBUG

	m_FrameToWriteTo = (m_FrameToWriteTo + 1) % 5;

	VkRenderPassBeginInfo beginInfo = {};
	beginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	beginInfo.renderPass = m_renderPass;
	beginInfo.framebuffer = m_historyFramebuffers[m_FrameToWriteTo]; // this is wrong, should not be using current frame
	beginInfo.renderArea.extent = context.extent;

	beginInfo.clearValueCount = 0;
	beginInfo.pClearValues = nullptr;

	VkViewport viewport{};
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = (float)context.extent.width;
	viewport.height = (float)context.extent.height;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;
	vkCmdSetViewport(cmd, 0, 1, &viewport);

	VkRect2D scissor{};
	scissor.offset = { 0,0 };
	scissor.extent = { context.extent.width, context.extent.height };
	vkCmdSetScissor(cmd, 0, 1, &scissor);

	vkCmdBeginRenderPass(cmd, &beginInfo, VK_SUBPASS_CONTENTS_INLINE);
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descriptorSets[currentFrame], 0, nullptr);

	// Draw large triangle here
	vkCmdDraw(cmd, 3, 1, 0, 0);

	vkCmdEndRenderPass(cmd);

#ifdef _DEBUG
	EndRenderPassLabel(cmd);
#endif
}

void vk::History::Destroy()
{
	for (size_t i = 0; i < 5; i++)
	{
		m_historyImages[i].Destroy(context.device);
	}
}

void vk::History::CreatePipeline()
{
	auto pipelineResult = vk::PipelineBuilder(context, PipelineType::GRAPHICS, VertexBinding::NONE, 0)
		.AddShader("assets/shaders/fs_tri.vert.spv", ShaderType::VERTEX)
		.AddShader("assets/shaders/history.frag.spv", ShaderType::FRAGMENT)
		.SetInputAssembly(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
		.SetDynamicState({ {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR} })
		.SetRasterizationState(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
		.SetPipelineLayout({ {m_descriptorSetLayout} })
		.SetSampling(VK_SAMPLE_COUNT_1_BIT)
		.AddBlendAttachmentState(VK_TRUE, VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK_BLEND_OP_ADD, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD)
		.SetDepthState(VK_FALSE, VK_FALSE, VK_COMPARE_OP_LESS_OR_EQUAL) // Turn depth read and write OFF ========
		.SetRenderPass(m_renderPass)
		.Build();

	m_pipeline = pipelineResult.first;
	m_pipelineLayout = pipelineResult.second;
}

void vk::History::CreateRenderPass()
{
	RenderPass builder(context.device, 1);

	m_renderPass = builder
		.AddAttachment(VK_FORMAT_R32G32B32A32_SFLOAT, VK_SAMPLE_COUNT_1_BIT, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
		.AddColorAttachmentRef(0, 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)

		// External -> 0 : Color
		.AddDependency(
			VK_SUBPASS_EXTERNAL, 0,
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
			VK_DEPENDENCY_BY_REGION_BIT
		)

		// 0 -> External : Color : Wait for color writing to finish on the attachment before the fragment shader tries to read from it
		.AddDependency(
			0, VK_SUBPASS_EXTERNAL,
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
			VK_DEPENDENCY_BY_REGION_BIT
		)

		.Build();

	context.SetObjectName(context.device, (uint64_t)m_renderPass, VK_OBJECT_TYPE_RENDER_PASS, "HistoryRenderPass");
}

void vk::History::CreateFramebuffers()
{
	for (size_t i = 0; i < 5; i++)
	{
		VkImageView imageView[1] = { m_historyImages[i].imageView };
		VkFramebufferCreateInfo fbInfo = {
			.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
			.renderPass = m_renderPass,
			.attachmentCount = static_cast<uint32_t>(1),
			.pAttachments = imageView,
			.width = m_width,
			.height = m_height,
			.layers = 1
		};

		VK_CHECK(vkCreateFramebuffer(context.device, &fbInfo, nullptr, &m_historyFramebuffers[i]), "Failed to create History pass framebuffer.");
	}
}

void vk::History::BuildDescriptors()
{
	m_descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);

	// Set = 0, binding 0 = rendered scene image
	std::vector<VkDescriptorSetLayoutBinding> bindings = {
		CreateDescriptorBinding(0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
	};

	m_descriptorSetLayout = CreateDescriptorSetLayout(context, bindings);

	AllocateDescriptorSets(context, context.descriptorPool, m_descriptorSetLayout, MAX_FRAMES_IN_FLIGHT, m_descriptorSets);

	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorImageInfo imgInfo = {
			.sampler = repeatSampler,
			.imageView = renderedImage.imageView,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL // VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		};

		UpdateDescriptorSet(context, 0, imgInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	}
}

