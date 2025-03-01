#include "Context.hpp"
#include "Composite.hpp"
#include "Pipeline.hpp"
#include "Utils.hpp"
#include "Buffer.hpp"
#include "RenderPass.hpp"

vk::Composite::Composite(Context& context, Image& LightingPass, Image& BloomPass, const std::vector<Image>& historyImages) :
	context{ context },
	LightingPass{ LightingPass },
	BloomPass{ BloomPass },
	historyImages{historyImages},
	m_Pipeline{ VK_NULL_HANDLE },
	m_PipelineLayout{ VK_NULL_HANDLE },
	m_descriptorSetLayout{ VK_NULL_HANDLE },
	m_renderPass{ VK_NULL_HANDLE },
	m_framebuffer{ VK_NULL_HANDLE },
	m_width{ 0 },
	m_height{ 0 }
{

	m_width = context.extent.width;
	m_height = context.extent.height;

	m_RenderTarget = CreateImageTexture2D(
		"CompositeRT",
		context,
		context.extent.width,
		context.extent.height,
		VK_FORMAT_R32G32B32A32_SFLOAT,
		VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		VK_IMAGE_ASPECT_COLOR_BIT,
		1
	);

	BuildDescriptors();
	CreateRenderPass();
	CreateFramebuffer();
	CreatePipeline();
}

vk::Composite::~Composite()
{
	m_RenderTarget.Destroy(context.device);

	vkDestroyPipeline(context.device, m_Pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, m_PipelineLayout, nullptr);

	vkDestroyFramebuffer(context.device, m_framebuffer, nullptr);
	vkDestroyRenderPass(context.device, m_renderPass, nullptr);
	vkDestroyDescriptorSetLayout(context.device, m_descriptorSetLayout, nullptr);
}

void vk::Composite::Resize()
{
	m_width = context.extent.width;
	m_height = context.extent.height;

	vkDestroyFramebuffer(context.device, m_framebuffer, nullptr);

	m_RenderTarget.Destroy(context.device);

	m_RenderTarget = CreateImageTexture2D(
		"CompositeRT",
		context,
		context.extent.width,
		context.extent.height,
		VK_FORMAT_R32G32B32A32_SFLOAT,
		VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		VK_IMAGE_ASPECT_COLOR_BIT,
		1
	);
	CreateFramebuffer();


	// DefLighting pass
	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorImageInfo imageInfo = {
			.sampler = repeatSamplerAniso,
			.imageView = LightingPass.imageView,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		};

		UpdateDescriptorSet(context, 0, imageInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	}

	// Bloom pass
	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorImageInfo imageInfo = {
			.sampler = repeatSamplerAniso,
			.imageView = BloomPass.imageView,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		};

		UpdateDescriptorSet(context, 1, imageInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	}
}


void vk::Composite::Execute(VkCommandBuffer cmd)
{

#ifdef _DEBUG
	RenderPassLabel(cmd, "Composite");
#endif // !DEBUG

	VkRenderPassBeginInfo rpBegin{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
	rpBegin.renderPass = m_renderPass;
	rpBegin.framebuffer = m_framebuffer;
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

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_PipelineLayout, 0, 1, &m_descriptorSets[currentFrame], 0, nullptr);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_Pipeline);
	vkCmdDraw(cmd, 3, 1, 0, 0);

	vkCmdEndRenderPass(cmd);

#ifdef _DEBUG
	EndRenderPassLabel(cmd);
#endif // !DEBUG
}

void vk::Composite::Update()
{

}

void vk::Composite::CreatePipeline()
{
	// Create the pipeline
	auto pipelineResult = vk::PipelineBuilder(context, PipelineType::GRAPHICS, VertexBinding::NONE, 0)
		.AddShader("assets/shaders/fs_tri.vert.spv", ShaderType::VERTEX)
		.AddShader("assets/shaders/composite.frag.spv", ShaderType::FRAGMENT)
		.SetInputAssembly(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
		.SetDynamicState({ {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR} })
		.SetRasterizationState(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
		.SetPipelineLayout({ {m_descriptorSetLayout} })
		.SetSampling(VK_SAMPLE_COUNT_1_BIT)
		.AddBlendAttachmentState()
		.SetDepthState(VK_FALSE, VK_FALSE, VK_COMPARE_OP_LESS_OR_EQUAL) // Turn depth read and write OFF ========
		.SetRenderPass(m_renderPass)
		.Build();

	m_Pipeline = pipelineResult.first;
	m_PipelineLayout = pipelineResult.second;
}

void vk::Composite::CreateRenderPass()
{
	RenderPass builder(context.device, 1);

	m_renderPass = builder
		.AddAttachment(VK_FORMAT_R32G32B32A32_SFLOAT, VK_SAMPLE_COUNT_1_BIT, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
		.AddColorAttachmentRef(0, 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
		// External -> 0 : Color
		.AddDependency(VK_SUBPASS_EXTERNAL, 0, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_DEPENDENCY_BY_REGION_BIT)

		// 0 -> External : Color : Wait for color writing to finish on the attachment before the fragment shader tries to read from it
		.AddDependency(0, VK_SUBPASS_EXTERNAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT, VK_DEPENDENCY_BY_REGION_BIT)
		.Build();

	context.SetObjectName(context.device, (uint64_t)m_renderPass, VK_OBJECT_TYPE_RENDER_PASS, "CompositeRenderPass");

}

void vk::Composite::CreateFramebuffer()
{
	std::vector<VkImageView> attachments = { m_RenderTarget.imageView };

	VkFramebufferCreateInfo fbcInfo = {
		.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
		.renderPass = m_renderPass,
		.attachmentCount = static_cast<uint32_t>(attachments.size()),
		.pAttachments = attachments.data(),
		.width = context.extent.width,
		.height = context.extent.height,
		.layers = 1
	};

	VK_CHECK(vkCreateFramebuffer(context.device, &fbcInfo, nullptr, &m_framebuffer), "Failed to create Composite pass framebuffer.");
}

void vk::Composite::BuildDescriptors()
{
	m_descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
	{
		// Set = 0, binding 0 = cameraUBO, binding = 1 = textures
		std::vector<VkDescriptorSetLayoutBinding> bindings = {
			CreateDescriptorBinding(0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT),
			CreateDescriptorBinding(1, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT),
			CreateDescriptorBinding(2, 5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT),
		};

		m_descriptorSetLayout = CreateDescriptorSetLayout(context, bindings);

		AllocateDescriptorSets(context, context.descriptorPool, m_descriptorSetLayout, MAX_FRAMES_IN_FLIGHT, m_descriptorSets);
	}

	// DefLighting pass
	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorImageInfo imageInfo = {
			.sampler = repeatSamplerAniso,
			.imageView = LightingPass.imageView,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		};

		UpdateDescriptorSet(context, 0, imageInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	}

	// Bloom pass
	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorImageInfo imageInfo = {
			.sampler = repeatSamplerAniso,
			.imageView = BloomPass.imageView,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		};

		UpdateDescriptorSet(context, 1, imageInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	}

	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		for (size_t image = 0; image < historyImages.size(); image++)
		{
			VkDescriptorImageInfo imgInfo = {
				.sampler = repeatSampler,
				.imageView = historyImages[image].imageView,
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL // VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
			};

			UpdateDescriptorSet(context, 2, imgInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
		}
	}
}
