#pragma once
#include <volk/volk.h>
#include <iostream>
#include <functional>
#include <glm/glm.hpp>
#include <array>

class Context;

#define VK_CHECK(call, message) \
	do { \
		VkResult result = call; \
		if (result != VK_SUCCESS) { \
			std::cout << "[VK ERROR]: " << message << " VkResult: " << result << std::endl; \
		} \
	} while (0)

#define ERROR(message) std::cout << "[ERROR]: " << message << std::endl; \

constexpr int NUM_LIGHTS = 51;

namespace vk
{
	enum class RenderType
	{
		FORWARD,
		RAYPASS,
		MESH_DENSITY
	};


	inline int MAX_FRAMES_IN_FLIGHT;
	inline int currentFrame;

	inline VkSampler repeatSamplerAniso;
	inline VkSampler repeatSampler;
	inline VkSampler clampToEdgeSamplerAniso;

	inline float maxAnisotropic;
	extern RenderType renderType;
}

namespace vk
{
	struct alignas(16) MeshPushConstants
	{
		glm::mat4 ModelMatrix;
		glm::vec4 BaseColourFactor;
		float Metallic;
		float Roughness;
	};

	struct LightUBO
	{
		alignas(4)	int type;
		alignas(16) glm::vec4 LightPosition;
		alignas(16) glm::vec4 LightColour;
		alignas(16) glm::mat4 LightSpaceMatrix;
	};

	struct PostProcessing
	{
		alignas(1) bool Enable;
	};

	struct LightBuffer
	{
		LightUBO lights[NUM_LIGHTS];
	};

	struct GuassianWeightsBuffer
	{
		float weights[22];
		float offsets[22];
	};

	struct RTX
	{
		alignas(4) int bounces;
		alignas(4) int frameIndex;
		alignas(4) int numPastFrames;
	};

	struct Reservoir
	{
		int Y; // Currently Y is index light array
		float Y_weight;
	};

	struct ReservoirStorage
	{
		Reservoir reservoirs[1280 * 720];
	};

	inline PostProcessing postProcessSettings = {};
	inline double deltaTime;
	inline uint32_t setRenderingPipeline = 1;
	inline uint32_t setAlphaMakingPipeline = 2;
	inline RTX rtxSettings = { 1, 0, 2 };
	inline VkDescriptorSetLayout materialDescriptorSetLayout;
	inline uint32_t frameNumber = 0;
	inline bool isAccumulating = false;
}

namespace vk
{
	void ExecuteSingleTimeCommands(Context& context, std::function<void(VkCommandBuffer)> recordCommands);

	// Sync
	void ImageBarrier(
		VkCommandBuffer cmd,
		VkImage img,
		VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask,
		VkImageLayout srcLayout, VkImageLayout dstLayout,
		VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask,
		VkImageSubresourceRange = VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
		uint32_t srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		uint32_t dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED);

	VkDescriptorSetLayout CreateDescriptorSetLayout(Context& context, const std::vector<VkDescriptorSetLayoutBinding>& bindings, VkDescriptorSetLayoutCreateFlags flags = 0);
	void AllocateDescriptorSets(Context& context, VkDescriptorPool descriptorPool, const VkDescriptorSetLayout descriptorLayout, uint32_t setCount, std::vector<VkDescriptorSet>& descriptorSet);
	void AllocateDescriptorSet(Context& context, VkDescriptorPool descriptorPool, const VkDescriptorSetLayout descriptorLayout, uint32_t setCount, VkDescriptorSet& descriptorSet);
	VkDescriptorSetLayoutBinding CreateDescriptorBinding(uint32_t binding, uint32_t count, VkDescriptorType type, VkShaderStageFlags shaderStage);

	// Update buffer descriptor
	void UpdateDescriptorSet(Context& context, uint32_t binding, VkDescriptorBufferInfo bufferInfo, VkDescriptorSet descriptorSet, VkDescriptorType descriptorType);
	// Update image descriptor
	void UpdateDescriptorSet(Context& context, uint32_t binding, VkDescriptorImageInfo imageInfo, VkDescriptorSet descriptorSet, VkDescriptorType descriptorType);

	// AS
	void UpdateDescriptorSet(Context& context, uint32_t binding, VkAccelerationStructureKHR AS, VkDescriptorSet descriptorSet, VkDescriptorType descriptorType);

	VkSampler CreateSampler(Context& context, VkSamplerAddressMode mode, VkBool32 EnableAnisotropic, VkCompareOp compareOp, VkFilter magFilter = VK_FILTER_LINEAR, VkFilter minFilter = VK_FILTER_LINEAR, VkSamplerMipmapMode samplerMipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR);

	// Don't use this. Needs to be removed.
	void BulkImageUpdate(Context& context, uint32_t binding, std::vector<VkDescriptorImageInfo> imageInfos, VkDescriptorSet descriptorSet, VkDescriptorType descriptorType);

	inline void RenderPassLabel(VkCommandBuffer commandBuffer, const char* labelName) {
		VkDebugUtilsLabelEXT label = {};
		label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
		label.pLabelName = labelName;
		vkCmdBeginDebugUtilsLabelEXT(commandBuffer, &label);
	}

	inline void EndRenderPassLabel(VkCommandBuffer commandBuffer) {
		vkCmdEndDebugUtilsLabelEXT(commandBuffer);
	}

	inline VkDeviceAddress GetBufferDeviceAddress(VkDevice device, VkBuffer buffer)
	{
		VkBufferDeviceAddressInfo addressInfo{
			.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
			.buffer = buffer
		};

		return vkGetBufferDeviceAddress(device, &addressInfo);
	}
}


/*
* Render normal geometry and then render foliage using different pipeline
* Color the base color into a metallic map
* Reference learnopengl camera movement and mouse look
* FIFO PRESENT MODE AND WAIT FOR V-SYNC
* Textues must be TRILINEAR (check sampler)
* Mouse movement should be activated on right mouse button click and deacitvated once right mouse is pressed again
TODO:
* - Lighting direction + color needs to be uniform pos [−0.2972,7.3100,−11.9532], color = [1,1,1]
* - BRDF
* - Different passes (Linear depth, PD, mip-maps[Disable anisotropic filtering for this task] etc)
* Turn mosiac on and off

// for the post-process
give ForwardPass it's own RT to render to (needs own render pass + framebuffer etc) - this pass will now make the depth buffer read  + write which can be used for final present pass to determine depth?
take RT in another pass as shader read only
do post process
write to swapchain image

*/


/*
	- The light needs to be placed somewhere so one light is across lighting deferred and shadow map
	- shadow frag coord needs to be pre-computed so its one single multiplication and a call to textureProj

	- Add point light to remaining braziers
	- Make sure to remove debug GLSL compiler flags -g -O0 in glslc.lua
*/