#include "Context.hpp"
#include "Scene.hpp"
#include "ShadowMap.hpp"
#include "Pipeline.hpp"
#include "Utils.hpp"
#include "Buffer.hpp"
#include "RenderPass.hpp"
#include "Camera.hpp"

#define RESOLUTION 1024
vk::ShadowMap::ShadowMap(Context& context, std::shared_ptr<Scene>& scene) : context{ context }, scene{ scene }
{
	assert(!scene->GetLights().empty());

	//128, 256, 512, 1024, 2048, 4096
	m_width = RESOLUTION;
	m_height = RESOLUTION;

	m_ShadowMap = CreateImageTexture2D(
		"ShadowMap_Depth_RT",
		context,
		m_width,
		m_height,
		VK_FORMAT_D32_SFLOAT,
		VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		VK_IMAGE_ASPECT_DEPTH_BIT,
		1
	);

	BuildDescriptors();
	CreateRenderPass();
	CreateFramebuffer();
	CreatePipeline();
}

vk::ShadowMap::~ShadowMap()
{
	m_ShadowMap.Destroy(context.device);
	vkDestroyPipeline(context.device, m_Pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, m_PipelineLayout, nullptr);

	vkDestroyFramebuffer(context.device, m_framebuffer, nullptr);
	vkDestroyRenderPass(context.device, m_renderPass, nullptr);
	vkDestroyDescriptorSetLayout(context.device, m_descriptorSetLayout, nullptr);
}

void vk::ShadowMap::Resize()
{
	vkDestroyFramebuffer(context.device, m_framebuffer, nullptr);
	m_ShadowMap.Destroy(context.device);

	m_ShadowMap = CreateImageTexture2D(
		"ShadowMap_Depth_RT",
		context,
		m_width,
		m_height,
		VK_FORMAT_D32_SFLOAT,
		VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		VK_IMAGE_ASPECT_DEPTH_BIT,
		1
	);

	CreateFramebuffer();
}

void vk::ShadowMap::Execute(VkCommandBuffer cmd)
{

#ifdef _DEBUG
	RenderPassLabel(cmd, "ShadowMap");
#endif 

	VkRenderPassBeginInfo beginInfo = {};
	beginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	beginInfo.renderPass = m_renderPass;
	beginInfo.framebuffer = m_framebuffer;
	beginInfo.renderArea.extent = { m_width, m_height };

	VkClearValue clearValues[1];
	clearValues[0].depthStencil.depth = { 1.0f };
	beginInfo.clearValueCount = 1;
	beginInfo.pClearValues = clearValues;

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

	vkCmdBeginRenderPass(cmd, &beginInfo, VK_SUBPASS_CONTENTS_INLINE);
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_Pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_PipelineLayout, 0, 1, &m_descriptorSets[currentFrame], 0, nullptr);

	scene->DrawGLTF(cmd, m_PipelineLayout);

	vkCmdEndRenderPass(cmd);

#ifdef _DEBUG
	EndRenderPassLabel(cmd);
#endif
}

void vk::ShadowMap::CreatePipeline()
{
	VkPushConstantRange pushConstantRange = {
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		.offset = 0,
		.size = sizeof(MeshPushConstants)
	};

	// Default pipeline
	auto ShadowMapPipelineRes = vk::PipelineBuilder(context, PipelineType::GRAPHICS, VertexBinding::BIND, 0)
		.AddShader("assets/shaders/shadow_map.vert.spv", ShaderType::VERTEX)
		.AddShader("assets/shaders/shadow_map.frag.spv", ShaderType::FRAGMENT)
		.SetInputAssembly(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
		.SetDynamicState({ {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR} })
		.SetRasterizationState(VK_POLYGON_MODE_FILL, VK_CULL_MODE_FRONT_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE)
		.SetPipelineLayout({ {m_descriptorSetLayout, materialDescriptorSetLayout} }, pushConstantRange)
		.SetSampling(VK_SAMPLE_COUNT_1_BIT)
		.AddBlendAttachmentState()
		.SetDepthState(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL)
		.SetRenderPass(m_renderPass)
		.Build();

	m_Pipeline = ShadowMapPipelineRes.first;
	m_PipelineLayout = ShadowMapPipelineRes.second;
}

void vk::ShadowMap::CreateRenderPass()
{
	RenderPass builder(context.device, 1);

	m_renderPass = builder
		.AddAttachment(VK_FORMAT_D32_SFLOAT, VK_SAMPLE_COUNT_1_BIT, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL)
		.SetDepthAttachmentRef(0, 0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)

		// External -> 0 : Depth ( Wait for previous depth writes to finish before writing )
		.AddDependency(VK_SUBPASS_EXTERNAL, 0, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_DEPENDENCY_BY_REGION_BIT)
		
		// 0 -> External : Depth ( Wait for depth to finish writing in this render pass before allowing other render passes to try and sample from it ) 
		.AddDependency(0, VK_SUBPASS_EXTERNAL, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT, VK_DEPENDENCY_BY_REGION_BIT)

		.Build();
}

void vk::ShadowMap::CreateFramebuffer()
{
	// Framebuffer
	std::vector<VkImageView> attachments = {
		m_ShadowMap.imageView
	};
	VkFramebufferCreateInfo fbcInfo = {
		.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
		.renderPass = m_renderPass,
		.attachmentCount = static_cast<uint32_t>(attachments.size()),
		.pAttachments = attachments.data(),
		.width = m_width,
		.height = m_height,
		.layers = 1
	};

	VK_CHECK(vkCreateFramebuffer(context.device, &fbcInfo, nullptr, &m_framebuffer), "Failed to create Forward pass framebuffer.");
}

void vk::ShadowMap::BuildDescriptors()
{
	m_descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
	{
		// Light UBO 
		std::vector<VkDescriptorSetLayoutBinding> bindings = {
			CreateDescriptorBinding(0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT), // SceneUBO (projection, view etc..)
		};

		m_descriptorSetLayout = CreateDescriptorSetLayout(context, bindings);
		AllocateDescriptorSets(context, context.descriptorPool, m_descriptorSetLayout, MAX_FRAMES_IN_FLIGHT, m_descriptorSets);
	}

	// Camera Transform UBO
	for (size_t i = 0; i < (size_t)MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorBufferInfo bufferInfo{};
		bufferInfo.buffer = scene->GetLightsUBO()[i].buffer;
		bufferInfo.offset = 0;
		bufferInfo.range = sizeof(LightUBO);
		UpdateDescriptorSet(context, 0, bufferInfo, m_descriptorSets[i], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
	}
}

void vk::ShadowMap::Update()
{

}

