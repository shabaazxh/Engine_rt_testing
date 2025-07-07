#include "Context.hpp"
#include "Pipeline.hpp"
#include "RenderPass.hpp"
#include "History.hpp"

vk::History::History(Context& context, const Image& renderedImage) : context{context}, renderedImage{renderedImage}
{
	m_FrameToWriteTo = currentFrame;
	m_width = context.extent.width;
	m_height = context.extent.height;

	m_rtxSettingsUBO.resize(MAX_FRAMES_IN_FLIGHT);
	for (auto& buffer : m_rtxSettingsUBO)
		buffer = CreateBuffer("HistoryRTXUBO", context, sizeof(RTX), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

	m_RenderTarget = CreateImageTexture2D(
		"History_Accum_RT",
		context,
		m_width,
		m_height,
		VK_FORMAT_R16G16B16A16_SFLOAT,
		VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
		VK_IMAGE_ASPECT_COLOR_BIT,
		1
	);

	ExecuteSingleTimeCommands(context, [&](VkCommandBuffer cmd) {

		ImageTransition(cmd, m_RenderTarget.image, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

	});

	BuildDescriptors();
	CreatePipeline();
}

vk::History::~History()
{
	for (auto& buffer : m_rtxSettingsUBO)
	{
		buffer.Destroy(context.device);
	}

	m_RenderTarget.Destroy(context.device);
	vkDestroyPipeline(context.device, m_pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, m_pipelineLayout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, m_descriptorSetLayout, nullptr);
}

void vk::History::Update()
{
	static uint32_t MaxFramesToAccumulate = 1000;

	if (isAccumulating)
	{
		rtxSettings.frameIndex++;

		if (rtxSettings.frameIndex >= MaxFramesToAccumulate)
		{
			rtxSettings.frameIndex = MaxFramesToAccumulate;
			std::printf("Completed accumulation %d\n", rtxSettings.frameIndex);
		}
	}
	else
	{
		rtxSettings.frameIndex = -1;
	}

	m_rtxSettingsUBO[currentFrame].WriteToBuffer(&rtxSettings, sizeof(RTX));
}

void vk::History::Resize()
{
	m_width = context.extent.width;
	m_height = context.extent.height;

	m_RenderTarget.Destroy(context.device);

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

void vk::History::Execute(VkCommandBuffer cmd)
{

#ifdef _DEBUG
	RenderPassLabel(cmd, "HistoryPass");
#endif // !DEBUG


	if (shouldClearBeforeDraw)
	{
		std::printf("Clearing History Render Target\n");
		// Transition image to transfer destination layout for clearing
		ImageTransition(
			cmd,
			m_RenderTarget.image,
			VK_FORMAT_R16G16B16A16_SFLOAT,
			VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			0, // srcAccessMask
			VK_ACCESS_TRANSFER_WRITE_BIT, // dstAccessMask
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, // srcStageMask (or the previous stage)
			VK_PIPELINE_STAGE_TRANSFER_BIT // dstStageMask
		);

		// Clear the image
		VkClearColorValue clearColor = {};
		clearColor.float32[0] = 0.0f;
		clearColor.float32[1] = 0.0f;
		clearColor.float32[2] = 0.0f;
		clearColor.float32[3] = 0.0f;

		VkImageSubresourceRange range = {};
		range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		range.baseMipLevel = 0;
		range.levelCount = 1;
		range.baseArrayLayer = 0;
		range.layerCount = 1;

		vkCmdClearColorImage(cmd, m_RenderTarget.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearColor, 1, &range);

	}

	// Transition to general layout for compute pass
	if (shouldClearBeforeDraw) {

		ImageTransition(
			cmd,
			m_RenderTarget.image,
			VK_FORMAT_R16G16B16A16_SFLOAT,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_IMAGE_LAYOUT_GENERAL,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
		);

		shouldClearBeforeDraw = false;
	}
	else
	{
		// Transition temporal acc image from shader read only to general layout to read and write to it using this compute pass
		ImageTransition(cmd, m_RenderTarget.image, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	}

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
		CreateDescriptorBinding(1, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT),
		CreateDescriptorBinding(2, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
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

	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorBufferInfo bufferInfo{};
		bufferInfo.buffer = m_rtxSettingsUBO[i].buffer;
		bufferInfo.offset = 0;
		bufferInfo.range = sizeof(RTX);
		UpdateDescriptorSet(context, 2, bufferInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
	}
}

