#include "MotionVectors.hpp"
#include "Pipeline.hpp"
#include "RenderPass.hpp"

vk::MotionVectors::MotionVectors(Context& context, Camera& camera, const Image& currentDepthBuffer)
	: context{context}, camera{camera}, currentDepthBuffer{currentDepthBuffer}
{
	m_width = context.extent.width;
	m_height = context.extent.height;
	m_previousCameraTransform = {};

	m_RenderTarget = CreateImageTexture2D(
		"MotionVectors_RT",
		context,
		m_width,
		m_height,
		VK_FORMAT_R16G16B16A16_SFLOAT,
		VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
		VK_IMAGE_ASPECT_COLOR_BIT,
		1
	);

	for (auto& buffer : m_previousCameraUB)
	{
		buffer = CreateBuffer("PreviousCameraUB", context, sizeof(CameraTransform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
	}

	CreateRenderPass();
	CreateFramebuffer();
	CreatePipeline();
}

vk::MotionVectors::~MotionVectors()
{
	m_RenderTarget.Destroy(context.device);
	vkDestroyPipeline(context.device, m_Pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, m_PipelineLayout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, m_DescriptorSetLayout, nullptr);
	vkDestroyFramebuffer(context.device, m_Framebuffer, nullptr);
	vkDestroyRenderPass(context.device, m_RenderPass, nullptr);
}

void vk::MotionVectors::Update()
{
	m_previousCameraTransform = camera.GetCameraTransform();
	m_previousCameraUB[currentFrame].WriteToBuffer(m_previousCameraTransform, sizeof(CameraTransform));
}

void vk::MotionVectors::Resize()
{
	m_width = context.extent.width;
	m_height = context.extent.height;
}

void vk::MotionVectors::Execute(VkCommandBuffer cmd)
{
#ifdef _DEBUG
	RenderPassLabel(cmd, "MotionVectors");
#endif // !DEBUG

	VkRenderPassBeginInfo rpBegin{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
	rpBegin.renderPass = m_RenderPass;
	rpBegin.framebuffer = m_Framebuffer;
	rpBegin.renderArea.extent = { m_width, m_height };

	VkClearValue clearValues[1];
	clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
	rpBegin.clearValueCount = 1;
	rpBegin.pClearValues = clearValues;

	VkViewport viewport{};
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = (float)m_width;
	viewport.height = (float)m_height;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;
	vkCmdSetViewport(cmd, 0, 1, &viewport);

	VkRect2D scissor{};
	scissor.offset = { 0,0 };
	scissor.extent = { m_width, m_height };
	vkCmdSetScissor(cmd, 0, 1, &scissor);

	vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_Pipeline);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_PipelineLayout, 0, 1, &m_DescriptorSets[currentFrame], 0, nullptr);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_Pipeline);
	vkCmdDraw(cmd, 3, 1, 0, 0);

	vkCmdEndRenderPass(cmd);

#ifdef _DEBUG
	EndRenderPassLabel(cmd);
#endif // !DEBUG
}

void vk::MotionVectors::CreatePipeline()
{
	auto pipelineResult = vk::PipelineBuilder(context, PipelineType::GRAPHICS, VertexBinding::NONE, 0)
		.AddShader("assets/shaders/fs_tri.vert.spv", ShaderType::VERTEX)
		.AddShader("assets/shaders/MotionVectors.frag.spv", ShaderType::FRAGMENT)
		.SetInputAssembly(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
		.SetDynamicState({ {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR} })
		.SetRasterizationState(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
		.SetPipelineLayout({ {m_DescriptorSetLayout} })
		.SetSampling(VK_SAMPLE_COUNT_1_BIT)
		.AddBlendAttachmentState()
		.SetDepthState(VK_FALSE, VK_FALSE, VK_COMPARE_OP_LESS_OR_EQUAL)
		.SetRenderPass(m_RenderPass)
		.Build();

	m_Pipeline = pipelineResult.first;
	m_PipelineLayout = pipelineResult.second;
}

void vk::MotionVectors::CreateRenderPass()
{
	RenderPass builder(context.device, 1);

	m_RenderPass = builder
		.AddAttachment(VK_FORMAT_R16G16B16A16_SFLOAT, VK_SAMPLE_COUNT_1_BIT, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
		.AddColorAttachmentRef(0, 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)

		// Ensure depth is written to before this pass tries to use it to sample from
		.AddDependency(VK_SUBPASS_EXTERNAL, 0, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT, VK_DEPENDENCY_BY_REGION_BIT)

		// Ensure this pass finishes writing to the motion vectors target before the next pass uses it to sample from in its fragment shader
		.AddDependency(0, VK_SUBPASS_EXTERNAL, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT, VK_DEPENDENCY_BY_REGION_BIT)
		.Build();

	context.SetObjectName(context.device, (uint64_t)m_RenderPass, VK_OBJECT_TYPE_RENDER_PASS, "MotionVectorsRenderPass");
}

void vk::MotionVectors::CreateFramebuffer()
{
	VkFramebufferCreateInfo fbInfo = {

		.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
		.renderPass = m_RenderPass,
		.attachmentCount = 1,
		.pAttachments = &m_RenderTarget.imageView,
		.width = m_width,
		.height = m_height,
		.layers = 1
	};

	VK_CHECK(vkCreateFramebuffer(context.device, &fbInfo, nullptr, &m_Framebuffer), "Failed to create Framebuffer for Motion Vectors");
}

void vk::MotionVectors::CreateDescriptors()
{
	m_DescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
	std::vector<VkDescriptorSetLayoutBinding> bindings = {

		CreateDescriptorBinding(0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT),
		CreateDescriptorBinding(1, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT),
		CreateDescriptorBinding(2, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT)
	};

	// Bind current frame depth buffer
	// Current frame camera uniform
	// Previous frame camera uniform

	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorImageInfo imgInfo = {
			.sampler = clampToEdgeSamplerAniso,
			.imageView = currentDepthBuffer.imageView,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		};
	}

	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorBufferInfo bufferInfo{};
		bufferInfo.buffer = camera.GetBuffers()[i].buffer;
		bufferInfo.offset = 0;
		bufferInfo.range = sizeof(CameraTransform);
		UpdateDescriptorSet(context, 1, bufferInfo, m_DescriptorSets[i], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
	}

	// Previous camera uniform
	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorBufferInfo bufferInfo{};
		bufferInfo.buffer = camera.GetBuffers()[i].buffer;
		bufferInfo.offset = 0;
		bufferInfo.range = sizeof(CameraTransform);
		UpdateDescriptorSet(context, 2, bufferInfo, m_DescriptorSets[i], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
	}

}
