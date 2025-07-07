// Image& initial_candidates, Image& hit_world_positions, Image& hit_normals, Image& motion_vectors

#include "Context.hpp"
#include "Camera.hpp"
#include "Scene.hpp"
#include "SpatialCompute.hpp"
#include "Pipeline.hpp"
#include "Utils.hpp"
#include "Buffer.hpp"

vk::SpatialCompute::SpatialCompute(Context& context, std::shared_ptr<Scene>& scene, std::shared_ptr<Camera>& camera, Image& initial_candidates, Image& temporal_pass_reservoirs, const GBuffer::GBufferMRT& gbufferMRT) :
	context{ context },
	scene{ scene },
	camera{ camera },
	initial_candidates{ initial_candidates },
	temporal_pass_reservoirs{ temporal_pass_reservoirs },
	gbufferMRT{ gbufferMRT },
	m_Pipeline{ VK_NULL_HANDLE },
	m_PipelineLayout{ VK_NULL_HANDLE },
	m_descriptorSetLayout{ VK_NULL_HANDLE },
	m_width{ 0 },
	m_height{ 0 }
{

	m_width = context.extent.width;
	m_height = context.extent.height;

	m_uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
	for (auto& buffer : m_uniformBuffers)
		buffer = CreateBuffer("SpatialComputeUBO", context, sizeof(uSpatialPass), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

	m_RenderTarget = CreateImageTexture2D(
		"SpatialComputeRT",
		context,
		m_width,
		m_height,
		VK_FORMAT_R32G32B32A32_SFLOAT,
		VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
		VK_IMAGE_ASPECT_COLOR_BIT,
		1
	);

	m_TemporaryShadingResult = CreateImageTexture2D(
		"TempShadingSpatialComputeRT",
		context,
		m_width,
		m_height,
		VK_FORMAT_R32G32B32A32_SFLOAT,
		VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
		VK_IMAGE_ASPECT_COLOR_BIT,
		1
	);


	ExecuteSingleTimeCommands(context, [&](VkCommandBuffer cmd) {

		ImageTransition(
			cmd,
			m_RenderTarget.image,
			VK_FORMAT_R32G32B32A32_SFLOAT,
			VK_IMAGE_LAYOUT_UNDEFINED,
			VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0,
			VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
		);

		ImageTransition(
			cmd,
			m_TemporaryShadingResult.image,
			VK_FORMAT_R32G32B32A32_SFLOAT,
			VK_IMAGE_LAYOUT_UNDEFINED,
			VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0,
			VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
		);

	});

	BuildDescriptors();
	CreatePipeline();
}

vk::SpatialCompute::~SpatialCompute()
{
	for (auto& buffer : m_uniformBuffers)
	{
		buffer.Destroy(context.device);
	}
	m_RenderTarget.Destroy(context.device);
	m_TemporaryShadingResult.Destroy(context.device);
	vkDestroyPipeline(context.device, m_Pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, m_PipelineLayout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, m_descriptorSetLayout, nullptr);
}

void vk::SpatialCompute::Resize()
{
	m_RenderTarget.Destroy(context.device);
	m_TemporaryShadingResult.Destroy(context.device);

	m_width = context.extent.width;
	m_height = context.extent.height;

	m_RenderTarget = CreateImageTexture2D(
		"SpatialComputeRT",
		context,
		m_width,
		m_height,
		VK_FORMAT_R32G32B32A32_SFLOAT,
		VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
		VK_IMAGE_ASPECT_COLOR_BIT,
		1
	);

	m_TemporaryShadingResult = CreateImageTexture2D(
		"TempShadingSpatialComputeRT",
		context,
		m_width,
		m_height,
		VK_FORMAT_R32G32B32A32_SFLOAT,
		VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
		VK_IMAGE_ASPECT_COLOR_BIT,
		1
	);


	ExecuteSingleTimeCommands(context, [&](VkCommandBuffer cmd) {

		ImageTransition(
			cmd,
			m_RenderTarget.image,
			VK_FORMAT_R32G32B32A32_SFLOAT,
			VK_IMAGE_LAYOUT_UNDEFINED,
			VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0,
			VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
		);

		ImageTransition(
			cmd,
			m_TemporaryShadingResult.image,
			VK_FORMAT_R32G32B32A32_SFLOAT,
			VK_IMAGE_LAYOUT_UNDEFINED,
			VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0,
			VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
		);

	});

	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorImageInfo imageInfo = {

			.sampler = clampToEdgeSamplerAniso,
			.imageView = initial_candidates.imageView,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		};

		UpdateDescriptorSet(context, 2, imageInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	}

	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorImageInfo imageInfo = {

			.sampler = clampToEdgeSamplerAniso,
			.imageView = temporal_pass_reservoirs.imageView,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		};

		UpdateDescriptorSet(context, 3, imageInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	}


	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorImageInfo imageInfo = {
			.sampler = VK_NULL_HANDLE,
			.imageView = m_RenderTarget.imageView,
			.imageLayout = VK_IMAGE_LAYOUT_GENERAL
		};

		UpdateDescriptorSet(context, 4, imageInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
	}

	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorImageInfo imageInfo = {
			.sampler = VK_NULL_HANDLE,
			.imageView = m_TemporaryShadingResult.imageView,
			.imageLayout = VK_IMAGE_LAYOUT_GENERAL
		};

		UpdateDescriptorSet(context, 5, imageInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
	}

	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		UpdateDescriptorSet(context, 6, scene->TopLevelAccelerationStructure.handle, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);
	}

	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorImageInfo imageInfo = {

			.sampler = clampToEdgeSamplerAniso,
			.imageView = gbufferMRT.WorldPositions.imageView,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		};

		UpdateDescriptorSet(context, 7, imageInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	}

	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorImageInfo imageInfo = {

			.sampler = clampToEdgeSamplerAniso,
			.imageView = gbufferMRT.Normal.imageView,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		};

		UpdateDescriptorSet(context, 8, imageInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	}

	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorImageInfo imageInfo = {

			.sampler = clampToEdgeSamplerAniso,
			.imageView = gbufferMRT.Albedo.imageView,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		};

		UpdateDescriptorSet(context, 9, imageInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	}


}

void vk::SpatialCompute::Execute(VkCommandBuffer cmd)
{
#ifdef _DEBUG
	RenderPassLabel(cmd, "SpatialCompute");
#endif // !DEBUG

	ImageTransition(
		cmd,
		m_RenderTarget.image,
		VK_FORMAT_R32G32B32A32_SFLOAT,
		VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
		VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
		VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
	);

	ImageTransition(
		cmd,
		m_TemporaryShadingResult.image,
		VK_FORMAT_R32G32B32A32_SFLOAT,
		VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
		VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
		VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
	);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_Pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 1, &m_descriptorSets[currentFrame], 0, nullptr);

	// 8x8x1 threads per dispatch
	vkCmdDispatch(cmd, m_width / 8, m_height / 8, 1);

	ImageTransition(
		cmd,
		m_RenderTarget.image,
		VK_FORMAT_R32G32B32A32_SFLOAT,
		VK_IMAGE_LAYOUT_GENERAL,
		VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
	);

	ImageTransition(
		cmd,
		m_TemporaryShadingResult.image,
		VK_FORMAT_R32G32B32A32_SFLOAT,
		VK_IMAGE_LAYOUT_GENERAL,
		VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
	);

#ifdef _DEBUG
	EndRenderPassLabel(cmd);
#endif // !DEBUG

}

void vk::SpatialCompute::Update()
{
	SpatialPassData.frameIndex = frameNumber;
	SpatialPassData.viewportSize = { m_width, m_height };
	SpatialPassData.M = SpatialPassData.M;
	SpatialPassData.radius = SpatialPassData.radius;
	m_uniformBuffers[currentFrame].WriteToBuffer(&SpatialPassData, sizeof(uSpatialPass));
}

void vk::SpatialCompute::CreatePipeline()
{
	auto pipelineResult = vk::PipelineBuilder(context, PipelineType::COMPUTE, VertexBinding::NONE, 0)
		.AddShader("assets/shaders/SpatialCompute.comp.spv", ShaderType::COMPUTE)
		.SetPipelineLayout({ {m_descriptorSetLayout} })
		.Build();

	m_Pipeline = pipelineResult.first;
	m_PipelineLayout = pipelineResult.second;

}

void vk::SpatialCompute::BuildDescriptors()
{
	m_descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
	{
		std::vector<VkDescriptorSetLayoutBinding> bindings = {
			CreateDescriptorBinding(0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT), // ubo
			CreateDescriptorBinding(1, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT), // Light ubo
			CreateDescriptorBinding(2, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT), // Initial candidates
			CreateDescriptorBinding(3, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT), // Temporal pass results
			CreateDescriptorBinding(4, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT), // Store spatial reuse updated reservoirs
			CreateDescriptorBinding(5, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT),  // Store shading result
			CreateDescriptorBinding(6, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_COMPUTE_BIT), // TLAS
			CreateDescriptorBinding(7, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT), // GBuffer - World position
			CreateDescriptorBinding(8, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT), // GBuffer - World Normal
			CreateDescriptorBinding(9, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT)  // GBuffer - Albedo
		};

		m_descriptorSetLayout = CreateDescriptorSetLayout(context, bindings);
		AllocateDescriptorSets(context, context.descriptorPool, m_descriptorSetLayout, MAX_FRAMES_IN_FLIGHT, m_descriptorSets);
	}

	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorBufferInfo buffer_info = {
			.buffer = m_uniformBuffers[i].buffer,
			.offset = 0,
			.range = sizeof(uSpatialPass)
		};
		UpdateDescriptorSet(context, 0, buffer_info, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
	}

	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorBufferInfo buffer_info = {
			.buffer = scene->GetLightsUBO()[i].buffer,
			.offset = 0,
			.range = sizeof(LightBuffer)
		};
		UpdateDescriptorSet(context, 1, buffer_info, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
	}

	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorImageInfo imageInfo = {

			.sampler = clampToEdgeSamplerAniso,
			.imageView = initial_candidates.imageView,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		};

		UpdateDescriptorSet(context, 2, imageInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	}

	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorImageInfo imageInfo = {

			.sampler = clampToEdgeSamplerAniso,
			.imageView = temporal_pass_reservoirs.imageView,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		};

		UpdateDescriptorSet(context, 3, imageInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	}


	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorImageInfo imageInfo = {
			.sampler = VK_NULL_HANDLE,
			.imageView = m_RenderTarget.imageView,
			.imageLayout = VK_IMAGE_LAYOUT_GENERAL
		};

		UpdateDescriptorSet(context, 4, imageInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
	}

	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorImageInfo imageInfo = {
			.sampler = VK_NULL_HANDLE,
			.imageView = m_TemporaryShadingResult.imageView,
			.imageLayout = VK_IMAGE_LAYOUT_GENERAL
		};

		UpdateDescriptorSet(context, 5, imageInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
	}

	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		UpdateDescriptorSet(context, 6, scene->TopLevelAccelerationStructure.handle, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);
	}

	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorImageInfo imageInfo = {

			.sampler = clampToEdgeSamplerAniso,
			.imageView = gbufferMRT.WorldPositions.imageView,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		};

		UpdateDescriptorSet(context, 7, imageInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	}

	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorImageInfo imageInfo = {

			.sampler = clampToEdgeSamplerAniso,
			.imageView = gbufferMRT.Normal.imageView,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		};

		UpdateDescriptorSet(context, 8, imageInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	}

	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorImageInfo imageInfo = {

			.sampler = clampToEdgeSamplerAniso,
			.imageView = gbufferMRT.Albedo.imageView,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		};

		UpdateDescriptorSet(context, 9, imageInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	}
}
