#include "Context.hpp"
#include "Utils.hpp"

namespace vk
{
	RenderType renderType = RenderType::FORWARD;
}

void vk::ExecuteSingleTimeCommands(Context& context, std::function<void(VkCommandBuffer)> recordCommands)
{
	VkCommandBufferAllocateInfo allocateCmd = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = context.transientCommandPool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1
	};

	VkCommandBuffer cmd = VK_NULL_HANDLE;
	VK_CHECK(vkAllocateCommandBuffers(context.device, &allocateCmd, &cmd), "Failed to allocate command buffer during one time submit.");

	VkCommandBufferBeginInfo beginInfo = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
	};

	// Create the fence
	VkFenceCreateInfo fenceInfo{
		.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
		.flags = VK_FENCE_CREATE_SIGNALED_BIT
	};

	VkFence fence = VK_NULL_HANDLE;
	VK_CHECK(vkCreateFence(context.device, &fenceInfo, nullptr, &fence), "Failed to create Fence.");

	vkResetFences(context.device, 1, &fence);

	vkBeginCommandBuffer(cmd, &beginInfo);
	recordCommands(cmd);
	vkEndCommandBuffer(cmd);

	VkSubmitInfo submitInfo = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers = &cmd
	};

	vkQueueSubmit(context.graphicsQueue, 1, &submitInfo, fence);
	vkQueueWaitIdle(context.graphicsQueue);

	vkFreeCommandBuffers(context.device, context.transientCommandPool, 1, &cmd);
	vkDestroyFence(context.device, fence, nullptr);

	vkResetCommandPool(context.device, context.transientCommandPool, 0);
}

void vk::ImageBarrier(
	VkCommandBuffer cmd,
	VkImage img,
	VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask,
	VkImageLayout srcLayout, VkImageLayout dstLayout,
	VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask,
	VkImageSubresourceRange subresourceRange,
	uint32_t srcQueueFamilyIndex,
	uint32_t dstQueueFamilyIndex)
{
	VkImageMemoryBarrier imgBarrier = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = srcAccessMask,
		.dstAccessMask = dstAccessMask,
		.oldLayout = srcLayout,
		.newLayout = dstLayout,
		.srcQueueFamilyIndex = srcQueueFamilyIndex,
		.dstQueueFamilyIndex = dstQueueFamilyIndex,
		.image = img,
		.subresourceRange = subresourceRange
	};

	vkCmdPipelineBarrier(cmd, srcStageMask, dstStageMask, 0, 0, nullptr, 0, nullptr, 1, &imgBarrier);
}

VkDescriptorSetLayout vk::CreateDescriptorSetLayout(vk::Context& context, const std::vector<VkDescriptorSetLayoutBinding>& bindings, VkDescriptorSetLayoutCreateFlags flags)
{
	VkDescriptorSetLayoutCreateInfo info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
	info.bindingCount = static_cast<uint32_t>(bindings.size());
	info.pBindings = bindings.data();
	info.flags = flags;

	VkDescriptorSetLayout layout = VK_NULL_HANDLE;
	VK_CHECK(vkCreateDescriptorSetLayout(context.device, &info, nullptr, &layout), "Failed to crate descriptor set layout");

	return layout;
}

void vk::AllocateDescriptorSets(Context& context, VkDescriptorPool descriptorPool, const VkDescriptorSetLayout descriptorLayout, uint32_t setCount, std::vector<VkDescriptorSet>& descriptorSet)
{
	std::vector<VkDescriptorSetLayout> setLayout(vk::MAX_FRAMES_IN_FLIGHT, descriptorLayout);

	VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
	allocInfo.descriptorPool = descriptorPool;
	allocInfo.descriptorSetCount = setCount;
	allocInfo.pSetLayouts = setLayout.data();

	descriptorSet.resize(vk::MAX_FRAMES_IN_FLIGHT);

	VK_CHECK(vkAllocateDescriptorSets(context.device, &allocInfo, descriptorSet.data()), "Failed to allocate descriptor sets");
}

void vk::AllocateDescriptorSet(Context& context, VkDescriptorPool descriptorPool, const VkDescriptorSetLayout descriptorLayout, uint32_t setCount, VkDescriptorSet& descriptorSet)
{
	std::vector<VkDescriptorSetLayout> setLayout(1, descriptorLayout);

	VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
	allocInfo.descriptorPool = descriptorPool;
	allocInfo.descriptorSetCount = setCount;
	allocInfo.pSetLayouts = setLayout.data();

	VK_CHECK(vkAllocateDescriptorSets(context.device, &allocInfo, &descriptorSet), "Failed to allocate descriptor sets");
}


// Define a descriptor set layout binding
// binding = which index slot to bind to e.g. 0,1,2..
// type = what is the layout binding type e.g. uniform or sampler
// shadeStage = which shader will this be accessed and used in? e.g. vertex, fragment or both
VkDescriptorSetLayoutBinding vk::CreateDescriptorBinding(uint32_t binding, uint32_t count, VkDescriptorType type, VkShaderStageFlags shaderStage)
{
	VkDescriptorSetLayoutBinding bindingLayout = {};
	bindingLayout.binding = binding;
	bindingLayout.descriptorType = type;
	bindingLayout.descriptorCount = count;
	bindingLayout.stageFlags = shaderStage;

	return bindingLayout;
}

// Update buffer descriptor
void vk::UpdateDescriptorSet(Context& context, uint32_t binding, VkDescriptorBufferInfo bufferInfo, VkDescriptorSet descriptorSet, VkDescriptorType descriptorType)
{
	VkWriteDescriptorSet descriptorWrite{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
	descriptorWrite.dstSet = descriptorSet;
	descriptorWrite.dstBinding = binding;
	descriptorWrite.dstArrayElement = 0;
	descriptorWrite.descriptorType = descriptorType;
	descriptorWrite.descriptorCount = 1;
	descriptorWrite.pBufferInfo = &bufferInfo;

	vkUpdateDescriptorSets(context.device, 1, &descriptorWrite, 0, nullptr);
}

// Update image descriptor
void vk::UpdateDescriptorSet(Context& context, uint32_t binding, VkDescriptorImageInfo imageInfo, VkDescriptorSet descriptorSet, VkDescriptorType descriptorType)
{
	VkWriteDescriptorSet descriptorWrite{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
	descriptorWrite.dstSet = descriptorSet;
	descriptorWrite.dstBinding = binding;
	descriptorWrite.dstArrayElement = 0;
	descriptorWrite.descriptorType = descriptorType;
	descriptorWrite.descriptorCount = 1;
	descriptorWrite.pImageInfo = &imageInfo;

	vkUpdateDescriptorSets(context.device, 1, &descriptorWrite, 0, nullptr);
}

void vk::UpdateDescriptorSet(Context& context, uint32_t binding, VkAccelerationStructureKHR AS, VkDescriptorSet descriptorSet, VkDescriptorType descriptorType)
{
	VkWriteDescriptorSetAccelerationStructureKHR descriptorAccelerationStructureInfo = {};
	descriptorAccelerationStructureInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
	descriptorAccelerationStructureInfo.accelerationStructureCount = 1;
	descriptorAccelerationStructureInfo.pAccelerationStructures = &AS;

	VkWriteDescriptorSet descriptorWrite{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
	descriptorWrite.dstSet = descriptorSet;
	descriptorWrite.dstBinding = binding;
	descriptorWrite.dstArrayElement = 0;
	descriptorWrite.descriptorType = descriptorType;
	descriptorWrite.descriptorCount = 1;
	descriptorWrite.pNext = &descriptorAccelerationStructureInfo;

	vkUpdateDescriptorSets(context.device, 1, &descriptorWrite, 0, nullptr);
}

VkSampler vk::CreateSampler(Context& context, VkSamplerAddressMode mode, VkBool32 EnableAnisotropic, VkCompareOp compareOp,
	VkFilter magFilter, VkFilter minFilter, VkSamplerMipmapMode samplerMipmapMode)
{
	VkSamplerCreateInfo samplerInfo{};
	samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerInfo.magFilter = magFilter;
	samplerInfo.minFilter = minFilter;
	samplerInfo.mipmapMode = samplerMipmapMode;
	samplerInfo.addressModeU = mode;
	samplerInfo.addressModeV = mode;
	samplerInfo.addressModeW = mode;
	samplerInfo.minLod = 0.0f;
	samplerInfo.maxLod = VK_LOD_CLAMP_NONE;
	samplerInfo.mipLodBias = 0.f;
	samplerInfo.maxAnisotropy = EnableAnisotropic ? maxAnisotropic : 1; // set max anisotropic supported by device if being used else set it to default 1
	samplerInfo.anisotropyEnable = EnableAnisotropic;
	samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
	samplerInfo.compareEnable = VK_TRUE;
	samplerInfo.compareOp = compareOp; // VK_COMPARE_OP_LESS_OR_EQUAL

	VkSampler sampler = VK_NULL_HANDLE;
	VK_CHECK(vkCreateSampler(context.device, &samplerInfo, nullptr, &sampler), "Failed to create sampler");

	return sampler;
}

void vk::BulkImageUpdate(Context& context, uint32_t binding, std::vector<VkDescriptorImageInfo> imageInfos, VkDescriptorSet descriptorSet, VkDescriptorType descriptorType)
{
	VkWriteDescriptorSet descriptorWrite{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
	descriptorWrite.dstSet = descriptorSet;
	descriptorWrite.dstBinding = binding;
	descriptorWrite.dstArrayElement = 0;
	descriptorWrite.descriptorType = descriptorType;
	descriptorWrite.descriptorCount = static_cast<uint32_t>(imageInfos.size());
	descriptorWrite.pImageInfo = imageInfos.data();

	vkUpdateDescriptorSets(context.device, 1, &descriptorWrite, 0, nullptr);
}
