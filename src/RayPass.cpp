#include "Context.hpp"
#include "Camera.hpp"
#include "Scene.hpp"
#include "RayPass.hpp"
#include "Pipeline.hpp"
#include "Utils.hpp"
#include "Buffer.hpp"
#include "RenderPass.hpp"

namespace
{
	uint32_t aligned_size(uint32_t value, uint32_t alignment)
	{
		return (value + alignment - 1) & ~(alignment - 1);
	}
}

vk::RayPass::RayPass(Context& context, std::shared_ptr<Scene>& scene, std::shared_ptr<Camera>& camera) :
	context{ context },
	scene{scene},
	camera{camera},
	m_Pipeline{ VK_NULL_HANDLE },
	m_PipelineLayout{ VK_NULL_HANDLE },
	m_descriptorSetLayout{ VK_NULL_HANDLE },
	m_width{ 0 },
	m_height{ 0 }
{

	m_rtxSettingsUBO.resize(MAX_FRAMES_IN_FLIGHT);
	for (auto& buffer : m_rtxSettingsUBO)
		buffer = CreateBuffer("RTXUBO", context, sizeof(RTX), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);


	m_width = context.extent.width;
	m_height = context.extent.height;

	m_RenderTarget = CreateImageTexture2D(
		"RayPassRT",
		context,
		m_width,
		m_height,
		VK_FORMAT_R32G32B32A32_SFLOAT,
		VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
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
	});

	BuildDescriptors();
	CreatePipeline();
}

vk::RayPass::~RayPass()
{

	for (auto& buffer : m_rtxSettingsUBO)
		buffer.Destroy(context.device);

	m_RenderTarget.Destroy(context.device);

	vkDestroyPipeline(context.device, m_Pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, m_PipelineLayout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, m_descriptorSetLayout, nullptr);

	RayGenShaderBindingTable->Destroy(context.device);
	MissShaderBindingTable->Destroy(context.device);
	HitShaderBindingTable->Destroy(context.device);
}

void vk::RayPass::Resize()
{
	m_width = context.extent.width;
	m_height = context.extent.height;

	m_RenderTarget.Destroy(context.device);

	m_RenderTarget = CreateImageTexture2D(
		"RayPassRT",
		context,
		context.extent.width,
		context.extent.height,
		VK_FORMAT_R32G32B32A32_SFLOAT,
		VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
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
	});

	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorImageInfo imageInfo = {
			.sampler = VK_NULL_HANDLE,
			.imageView = m_RenderTarget.imageView,
			.imageLayout = VK_IMAGE_LAYOUT_GENERAL
		};

		UpdateDescriptorSet(context, 1, imageInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
	}
}


void vk::RayPass::Execute(VkCommandBuffer cmd)
{

#ifdef _DEBUG
	RenderPassLabel(cmd, "RayPass");
#endif // !DEBUG

	VkPhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingPipelineProperties{};
	rayTracingPipelineProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;

	VkPhysicalDeviceProperties2 deviceProperties{};
	deviceProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
	deviceProperties.pNext = &rayTracingPipelineProperties;

	vkGetPhysicalDeviceProperties2(context.pDevice, &deviceProperties);

	// Get shader group handle size & alignment
	const uint32_t handle_size = rayTracingPipelineProperties.shaderGroupHandleSize;
	const uint32_t handle_size_aligned = aligned_size(handle_size, rayTracingPipelineProperties.shaderGroupHandleAlignment);
	const uint32_t handle_alignment = rayTracingPipelineProperties.shaderGroupHandleAlignment;
	const uint32_t maxRayRecursionDepth = rayTracingPipelineProperties.maxRayRecursionDepth;


	ImageTransition(
		cmd,
		m_RenderTarget.image,
		VK_FORMAT_R32G32B32A32_SFLOAT,
		VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
		VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
		VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR
	);

		VkStridedDeviceAddressRegionKHR raygen_shader_sbt_entry{};
		raygen_shader_sbt_entry.deviceAddress = GetBufferDeviceAddress(context.device, RayGenShaderBindingTable->buffer);
		raygen_shader_sbt_entry.stride = handle_size_aligned;
		raygen_shader_sbt_entry.size = handle_size_aligned;

		VkStridedDeviceAddressRegionKHR miss_shader_sbt_entry{};
		miss_shader_sbt_entry.deviceAddress = GetBufferDeviceAddress(context.device, MissShaderBindingTable->buffer);
		miss_shader_sbt_entry.stride = handle_size_aligned;
		miss_shader_sbt_entry.size = handle_size_aligned * 2;

		VkStridedDeviceAddressRegionKHR hit_shader_sbt_entry{};
		hit_shader_sbt_entry.deviceAddress = GetBufferDeviceAddress(context.device, HitShaderBindingTable->buffer);
		hit_shader_sbt_entry.stride = handle_size_aligned;
		hit_shader_sbt_entry.size = handle_size_aligned;

		VkStridedDeviceAddressRegionKHR callable_shader_sbt_entry{};

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_Pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_PipelineLayout, 0, 1, &m_descriptorSets[currentFrame], 0, nullptr);
	vkCmdTraceRaysKHR(cmd, &raygen_shader_sbt_entry, &miss_shader_sbt_entry, &hit_shader_sbt_entry, &callable_shader_sbt_entry, context.extent.width, context.extent.height, 3);

	ImageTransition(
		cmd,
		m_RenderTarget.image,
		VK_FORMAT_R32G32B32A32_SFLOAT,
		VK_IMAGE_LAYOUT_GENERAL,
		VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
		VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
	);

#ifdef _DEBUG
	EndRenderPassLabel(cmd);
#endif // !DEBUG
}

void vk::RayPass::Update()
{
	rtxSettings.bounces = rtxSettings.bounces;
	rtxSettings.frameIndex = (frameNumber);
	m_rtxSettingsUBO[currentFrame].WriteToBuffer(&rtxSettings, sizeof(RTX));
}

void vk::RayPass::CreatePipeline()
{
	// Create the pipeline
	auto pipelineResult = vk::PipelineBuilder(context, PipelineType::RAYTRACING, VertexBinding::NONE, 0)
		.AddShader("assets/shaders/raygen.rgen.spv", ShaderType::RAYGEN)
		.AddShader("assets/shaders/miss.rmiss.spv", ShaderType::MISS)
		.AddShader("assets/shaders/shadowmiss.rmiss.spv", ShaderType::MISS)
		.AddShader("assets/shaders/closesthit.rchit.spv", ShaderType::HIT)
		.AddShader("assets/shaders/diffusehit.rchit.spv", ShaderType::HIT)
		.AddShader("assets/shaders/shadowhit.rchit.spv", ShaderType::HIT)
		.CreateShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 0) // raygen
		.CreateShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 1) // miss
		.CreateShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 2) // shadow-miss
		.CreateShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR, VK_SHADER_UNUSED_KHR, 3) // regular hit
		.CreateShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR, VK_SHADER_UNUSED_KHR, 4) // diffuse hit
		.CreateShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR, VK_SHADER_UNUSED_KHR, 5) // diffuse hit
		.SetPipelineLayout({ {m_descriptorSetLayout} })
		.Build();

	m_Pipeline = pipelineResult.first;
	m_PipelineLayout = pipelineResult.second;

	CreateShaderBindingTable();
}

void vk::RayPass::CreateShaderBindingTable()
{
	// Now create the shader binding table
	VkPhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingPipelineProperties{};
	rayTracingPipelineProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
	VkPhysicalDeviceProperties2 deviceProperties = {};
	deviceProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
	deviceProperties.pNext = &rayTracingPipelineProperties;
	vkGetPhysicalDeviceProperties2(context.pDevice, &deviceProperties);

	const uint32_t           handle_size = rayTracingPipelineProperties.shaderGroupHandleSize;
	const uint32_t           handle_size_aligned = aligned_size(rayTracingPipelineProperties.shaderGroupHandleSize, rayTracingPipelineProperties.shaderGroupHandleAlignment);
	const uint32_t           handle_alignment = rayTracingPipelineProperties.shaderGroupHandleAlignment;
	const uint32_t           group_count = static_cast<uint32_t>(6);
	const uint32_t           sbt_size = group_count * handle_size_aligned;
	const VkBufferUsageFlags sbt_buffer_usage_flags = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
	const VmaMemoryUsage     sbt_memory_usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

	RayGenShaderBindingTable = std::make_unique<Buffer>(
		CreateBuffer(
			"RayGenShaderBindingTable",
			context,
			handle_size_aligned,
			sbt_buffer_usage_flags,
			VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
		)
	);

	MissShaderBindingTable = std::make_unique<Buffer>(
		CreateBuffer(
			"MissShaderBindingTable",
			context,
			handle_size_aligned * 2,
			sbt_buffer_usage_flags,
			VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
		)
	);

	HitShaderBindingTable = std::make_unique<Buffer>(
		CreateBuffer(
			"HitShaderBindingTable",
			context,
			handle_size_aligned * 3,
			sbt_buffer_usage_flags,
			VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
		)
	);


	std::vector<uint8_t> shaderHandleStorage(sbt_size);
	VK_CHECK(vkGetRayTracingShaderGroupHandlesKHR(context.device, m_Pipeline, 0, group_count, sbt_size, shaderHandleStorage.data()), "Failed to get shader group handles");

	RayGenShaderBindingTable->WriteToBuffer(shaderHandleStorage.data(), handle_size_aligned);
	MissShaderBindingTable->WriteToBuffer(shaderHandleStorage.data() + handle_size_aligned, handle_size_aligned * 2);
	HitShaderBindingTable->WriteToBuffer(shaderHandleStorage.data() + handle_size_aligned * 3, handle_size_aligned * 3);
}

void vk::RayPass::BuildDescriptors()
{
	m_descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
	{
		// Set = 0, binding 0 = cameraUBO, binding = 1 = textures
		std::vector<VkDescriptorSetLayoutBinding> bindings = {
			CreateDescriptorBinding(0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR),
			CreateDescriptorBinding(1, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR),
			CreateDescriptorBinding(2, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR),
			CreateDescriptorBinding(3, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR),
			CreateDescriptorBinding(4, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR),
			CreateDescriptorBinding(5, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR),
			CreateDescriptorBinding(6, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR),
			CreateDescriptorBinding(7, 300, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR),
			CreateDescriptorBinding(8, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR),
			CreateDescriptorBinding(9, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
		};

		m_descriptorSetLayout = CreateDescriptorSetLayout(context, bindings);

		AllocateDescriptorSets(context, context.descriptorPool, m_descriptorSetLayout, MAX_FRAMES_IN_FLIGHT, m_descriptorSets);
	}

	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		UpdateDescriptorSet(context, 0, scene->TopLevelAccelerationStructure.handle, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);
	}

	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorImageInfo imageInfo = {
			.sampler = VK_NULL_HANDLE,
			.imageView = m_RenderTarget.imageView,
			.imageLayout = VK_IMAGE_LAYOUT_GENERAL
		};

		UpdateDescriptorSet(context, 1, imageInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
	}

	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorBufferInfo bufferInfo{};
		bufferInfo.buffer = camera->GetBuffers()[i].buffer;
		bufferInfo.offset = 0;
		bufferInfo.range = sizeof(CameraTransform);
		UpdateDescriptorSet(context, 2, bufferInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
	}

	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorBufferInfo vertexBufferInfo{};
		vertexBufferInfo.buffer = scene->vertexBuffer.buffer;
		vertexBufferInfo.offset = 0;
		vertexBufferInfo.range = VK_WHOLE_SIZE;

		VkDescriptorBufferInfo indexBufferInfo{};
		indexBufferInfo.buffer = scene->indexBuffer.buffer;
		indexBufferInfo.offset = 0;
		indexBufferInfo.range = VK_WHOLE_SIZE;

		VkDescriptorBufferInfo offsetBufferInfo{};
		offsetBufferInfo.buffer = scene->meshOffsetBuffer.buffer;
		offsetBufferInfo.offset = 0;
		offsetBufferInfo.range = VK_WHOLE_SIZE;

		VkDescriptorBufferInfo materialBufferInfo{};
		materialBufferInfo.buffer = scene->RTMaterialsBuffer.buffer;
		materialBufferInfo.offset = 0;
		materialBufferInfo.range = VK_WHOLE_SIZE;

		UpdateDescriptorSet(context, 3, vertexBufferInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
		UpdateDescriptorSet(context, 4, indexBufferInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
		UpdateDescriptorSet(context, 5, offsetBufferInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
		UpdateDescriptorSet(context, 6, materialBufferInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	}

	std::vector<VkDescriptorImageInfo> imageInfos;


	for (size_t i = 0; i < scene->textures.size(); i++)
	{
		VkDescriptorImageInfo imgInfo = {
			.sampler = repeatSamplerAniso,
			.imageView = scene->textures[i].imageView,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		};

		imageInfos.push_back(imgInfo);
	}

	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		BulkImageUpdate(context, 7, imageInfos, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	}

	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorBufferInfo bufferInfo{};
		bufferInfo.buffer = scene->GetLightsUBO()[i].buffer;
		bufferInfo.offset = 0;
		bufferInfo.range = sizeof(LightBuffer);
		UpdateDescriptorSet(context, 8, bufferInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
	}

	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorBufferInfo bufferInfo{};
		bufferInfo.buffer = m_rtxSettingsUBO[i].buffer;
		bufferInfo.offset = 0;
		bufferInfo.range = sizeof(RTX);
		UpdateDescriptorSet(context, 9, bufferInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
	}
}
