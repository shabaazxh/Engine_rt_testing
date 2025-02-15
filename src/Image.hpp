#pragma once
#include <volk/volk.h>
#include <vk_mem_alloc.h>
#include <string>


namespace vk
{
	class Context;

	class Image
	{
	public:
		Image() noexcept = default;
		Image(const std::string name, VmaAllocator allocator, VkImage image, VkImageView imageView, VmaAllocation allocation) noexcept;

		// Copy assignment and operator is deleted
		Image(const Image&) = delete;
		Image& operator=(const Image&) = delete;

		// Move assignment + operator is ok
		Image(Image&&) noexcept;
		Image& operator=(Image&&) noexcept;

		void Destroy(VkDevice device);

		std::string name;
		VmaAllocation allocation;
		VkImage image;
		VkImageView imageView;
		VmaAllocator allocator;
	};

	void ImageTransition(VkCommandBuffer cmd, VkImage image, VkFormat format, VkImageLayout currentLayout, VkImageLayout newLayout, VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask, VkPipelineStageFlagBits srcStageMask, VkPipelineStageFlagBits dstStageMask);
	uint32_t ComputeMipLevels(uint32_t width, uint32_t height);
	Image LoadTextureFromDisk(const std::string& path, Context& context, VkFormat format);
	Image CreateImageTexture2D(const std::string name, Context& context, uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags imageaspectFlags, uint32_t mipLevels = 1);
}
