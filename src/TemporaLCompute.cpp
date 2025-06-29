// Image& initial_candidates, Image& hit_world_positions, Image& hit_normals, Image& motion_vectors

#include "Context.hpp"
#include "Camera.hpp"
#include "Scene.hpp"
#include "TemporalCompute.hpp"
#include "Pipeline.hpp"
#include "Utils.hpp"
#include "Buffer.hpp"

vk::TemporalCompute::TemporalCompute(Context& context, std::shared_ptr<Scene>& scene, std::shared_ptr<Camera>& camera, Image& initial_candidates, Image& hit_world_positions, Image& hit_normals, Image& motion_vectors, const GBuffer::GBufferMRT& gbufferMRT) :
	context{ context },
	scene{ scene },
	camera{ camera },
	initial_candidates{ initial_candidates },
	motion_vectors{ motion_vectors },
	hit_world_positions { hit_world_positions },
	hit_normals { hit_normals },
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
	for(auto& buffer : m_uniformBuffers)
		buffer = CreateBuffer("TemporalComputeUBO", context, sizeof(uTemporalPass), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

	m_RenderTarget = CreateImageTexture2D(
		"TemporalComputeRT",
		context,
		m_width,
		m_height,
		VK_FORMAT_R32G32B32A32_SFLOAT,
		VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
		VK_IMAGE_ASPECT_COLOR_BIT,
		1
	);

	// This starts off as shader read only since it'll be read from during the TemporalCompute pass first
	// then at the end, we transition to transfer dst optimal and copy data into it which requires layout transition
	m_PreviousImage = CreateImageTexture2D(
		"PreviousImage",
		context,
		m_width,
		m_height,
		VK_FORMAT_R32G32B32A32_SFLOAT,
		VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
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
			VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
		);


		ImageTransition(
			cmd,
			m_PreviousImage.image,
			VK_FORMAT_R32G32B32A32_SFLOAT,
			VK_IMAGE_LAYOUT_UNDEFINED,
			VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0,
			VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
		);

		});

	BuildDescriptors();
	CreatePipeline();
}

void vk::TemporalCompute::CopyImageToImage(const Image& currentSpatialTemporalComputeReservoirImage)
{
	// Useful link: https://gpuopen.com/learn/vulkan-barriers-explained/
	ExecuteSingleTimeCommands(context, [&](VkCommandBuffer cmd)
		{
			// Transition the output to a TRANSFER_SRC_OPTIMAL layout since we will use this as the source
			// to copy data to the DST image
			// Write the reservoirs for each pixel to a seperate output image in the spatial reuse pass
			// Ensure its final layout is SHADER_READ_ONLY so this can transition it
			ImageTransition(
				cmd,
				currentSpatialTemporalComputeReservoirImage.image,
				VK_FORMAT_R32G32B32A32_SFLOAT,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				VK_ACCESS_SHADER_READ_BIT,
				VK_ACCESS_TRANSFER_READ_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, // Before this copy op, this image is sampled in a compute shader
				VK_PIPELINE_STAGE_TRANSFER_BIT
			);

			// Transition to TRANSFER_DST_OPTIMAL layout since it will data copied to it from the other image
			ImageTransition(
				cmd,
				m_PreviousImage.image,
				VK_FORMAT_R32G32B32A32_SFLOAT,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				VK_ACCESS_SHADER_READ_BIT,
				VK_ACCESS_TRANSFER_WRITE_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, // Before this copy op, this image is sampled in a compute shader
				VK_PIPELINE_STAGE_TRANSFER_BIT
			);

			VkImageCopy imgCopy =
			{
				.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
				.srcOffset = {0,0, 0},
				.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
				.dstOffset = {0, 0, 0},
				.extent = {m_width, m_height, 1}
			};

			vkCmdCopyImage(
				cmd,
				currentSpatialTemporalComputeReservoirImage.image,
				VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				m_PreviousImage.image,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				1,
				&imgCopy
			);


			// Transiton the resource back to make sure its in the correct layout for the rendering loop where it will
			// transition these resources to ensure they're in the correct layout for the ray tracing shaders to write to it

			ImageTransition(
				cmd,
				currentSpatialTemporalComputeReservoirImage.image,
				VK_FORMAT_R32G32B32A32_SFLOAT,
				VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				VK_ACCESS_TRANSFER_READ_BIT,
				VK_ACCESS_SHADER_READ_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
			);

			// Transition this back to shader read only for the same reasons as the m_RenderTarget transition back after finishing copy
			ImageTransition(
				cmd,
				m_PreviousImage.image,
				VK_FORMAT_R32G32B32A32_SFLOAT,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				VK_ACCESS_TRANSFER_WRITE_BIT,
				VK_ACCESS_SHADER_READ_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
			);
		});
}


vk::TemporalCompute::~TemporalCompute()
{
	for (auto& buffer : m_uniformBuffers)
	{
		buffer.Destroy(context.device);
	}
	m_RenderTarget.Destroy(context.device);
	m_PreviousImage.Destroy(context.device);
	vkDestroyPipeline(context.device, m_Pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, m_PipelineLayout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, m_descriptorSetLayout, nullptr);
}

void vk::TemporalCompute::Resize()
{

}

void vk::TemporalCompute::Execute(VkCommandBuffer cmd)
{
#ifdef _DEBUG
	RenderPassLabel(cmd, "TemporalCompute");
#endif // !DEBUG

	ImageTransition(
		cmd,
		m_RenderTarget.image,
		VK_FORMAT_R32G32B32A32_SFLOAT,
		VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
		VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
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
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
	);

#ifdef _DEBUG
	EndRenderPassLabel(cmd);
#endif // !DEBUG

}

void vk::TemporalCompute::Update()
{
	TemporalPassData.frameIndex = frameNumber;
	TemporalPassData.viewportSize = { m_width, m_height };
	TemporalPassData.M = TemporalPassData.M;
	m_uniformBuffers[currentFrame].WriteToBuffer(&TemporalPassData, sizeof(uTemporalPass));
}

void vk::TemporalCompute::CreatePipeline()
{
	auto pipelineResult = vk::PipelineBuilder(context, PipelineType::COMPUTE, VertexBinding::NONE, 0)
		.AddShader("assets/shaders/TemporalCompute.comp.spv", ShaderType::COMPUTE)
		.SetPipelineLayout({{m_descriptorSetLayout}})
		.Build();

	m_Pipeline = pipelineResult.first;
	m_PipelineLayout = pipelineResult.second;

}

void vk::TemporalCompute::BuildDescriptors()
{
	m_descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
	{
		std::vector<VkDescriptorSetLayoutBinding> bindings = {
			CreateDescriptorBinding(0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT), // ubo
			CreateDescriptorBinding(1, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT), // Light ubo
			CreateDescriptorBinding(2, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT), // Initial candidates
			CreateDescriptorBinding(3, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT), // World
			CreateDescriptorBinding(4, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT), // Normal
			CreateDescriptorBinding(5, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT), // Motion vectors
			CreateDescriptorBinding(6, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT), // Previous frame
			CreateDescriptorBinding(7, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT), // Output
			CreateDescriptorBinding(8, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_COMPUTE_BIT),
			CreateDescriptorBinding(9, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT),  // GBuffer - World position
			CreateDescriptorBinding(10, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT) // GBuffer - World Normal
		};

		m_descriptorSetLayout = CreateDescriptorSetLayout(context, bindings);
		AllocateDescriptorSets(context, context.descriptorPool, m_descriptorSetLayout, MAX_FRAMES_IN_FLIGHT, m_descriptorSets);
	}

	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorBufferInfo buffer_info = {
			.buffer = m_uniformBuffers[i].buffer,
			.offset = 0,
			.range = sizeof(uTemporalPass)
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
			.imageView = hit_world_positions.imageView,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		};

		UpdateDescriptorSet(context, 3, imageInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	}

	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorImageInfo imageInfo = {

			.sampler = clampToEdgeSamplerAniso,
			.imageView = hit_normals.imageView,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		};

		UpdateDescriptorSet(context, 4, imageInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	}

	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorImageInfo imageInfo = {

			.sampler = clampToEdgeSamplerAniso,
			.imageView = motion_vectors.imageView,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		};

		UpdateDescriptorSet(context, 5, imageInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	}

	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorImageInfo imageInfo = {

			.sampler = clampToEdgeSamplerAniso,
			.imageView = m_PreviousImage.imageView,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		};

		UpdateDescriptorSet(context, 6, imageInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	}

	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorImageInfo imageInfo = {
			.sampler = VK_NULL_HANDLE,
			.imageView = m_RenderTarget.imageView,
			.imageLayout = VK_IMAGE_LAYOUT_GENERAL
		};

		UpdateDescriptorSet(context, 7, imageInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
	}

	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		UpdateDescriptorSet(context, 8, scene->TopLevelAccelerationStructure.handle, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);
	}

	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorImageInfo imageInfo = {

			.sampler = clampToEdgeSamplerAniso,
			.imageView = gbufferMRT.WorldPositions.imageView,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		};

		UpdateDescriptorSet(context, 9, imageInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	}

	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorImageInfo imageInfo = {

			.sampler = clampToEdgeSamplerAniso,
			.imageView = gbufferMRT.Normal.imageView,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		};

		UpdateDescriptorSet(context, 10, imageInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	}
}
