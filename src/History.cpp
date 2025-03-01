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

	m_RenderTarget = CreateImageTexture2D(
		"History_Accum_RT",
		context,
		m_width,
		m_height,
		VK_FORMAT_R16G16B16A16_SFLOAT,
		VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		VK_IMAGE_ASPECT_COLOR_BIT,
		1
	);

	ExecuteSingleTimeCommands(context, [&](VkCommandBuffer cmd) {

		ImageTransition(cmd, m_RenderTarget.image, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

	});

	BuildDescriptors();
	CreatePipeline();
}


void vk::History::Execute(VkCommandBuffer cmd)
{

#ifdef _DEBUG
	RenderPassLabel(cmd, "HistoryPass");
#endif // !DEBUG

	// Transition temporal acc image from shader read only to general layout to read and write to it using this compute pass
	ImageTransition(cmd, m_RenderTarget.image, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

	// Execute horizontal blur
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &m_descriptorSets[currentFrame], 0, nullptr);
	vkCmdDispatch(cmd, m_width / 16, m_height / 16, 1);

	// Transition temporal acc image to shader read only to be used later
	ImageTransition(cmd, m_RenderTarget.image, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

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
	// TODO: need to destroy framebuffer, renderpass, pipeline etc
}

void vk::History::CreatePipeline()
{
	auto pipelineResult = vk::PipelineBuilder(context, PipelineType::COMPUTE, VertexBinding::NONE, 0)
		.AddShader("assets/shaders/history_compute.comp.spv", ShaderType::COMPUTE)
		.SetPipelineLayout({ m_descriptorSetLayout })
		.Build();

	m_pipeline = pipelineResult.first;
	m_pipelineLayout = pipelineResult.second;
}

void vk::History::BuildDescriptors()
{
	m_descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);

	// Set = 0, binding 0 = rendered scene image
	std::vector<VkDescriptorSetLayoutBinding> bindings = {
		CreateDescriptorBinding(0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT),
		CreateDescriptorBinding(1, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT)
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

	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorImageInfo imgInfo = {
			.sampler = VK_NULL_HANDLE,
			.imageView = m_RenderTarget.imageView,
			.imageLayout = VK_IMAGE_LAYOUT_GENERAL
		};

		UpdateDescriptorSet(context, 1, imgInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
	}
}

