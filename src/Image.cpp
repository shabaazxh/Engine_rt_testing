#include "Context.hpp"
#include "Image.hpp"
#include "Utils.hpp"
#include "Buffer.hpp"
#include "stb_image.h"
#include <assert.h>


vk::Image::Image(const std::string name, VmaAllocator allocator, VkImage image, VkImageView imageView, VmaAllocation allocation) noexcept :
	name{name}, allocation{allocation}, image{image}, imageView{imageView}, allocator{allocator} {}


vk::Image::Image(Image&& other) noexcept :
	name(std::exchange(other.name, "")),
	allocation(std::exchange(other.allocation, VK_NULL_HANDLE)),
	image(std::exchange(other.image, VK_NULL_HANDLE)),
	imageView(std::exchange(other.imageView, VK_NULL_HANDLE)),
	allocator(std::exchange(other.allocator, VK_NULL_HANDLE)) {

	//std::cout << "Move Constructing Image\n";
}


vk::Image& vk::Image::operator=(vk::Image&& other) noexcept
{
	std::swap(name, other.name);
	std::swap(allocation, other.allocation);
	std::swap(image, other.image);
	std::swap(imageView, other.imageView);
	std::swap(allocator, other.allocator);

	return *this;
}

void vk::Image::Destroy(VkDevice device)
{
	if (image != VK_NULL_HANDLE)
	{
		assert(allocator != VK_NULL_HANDLE);
		assert(allocation != VK_NULL_HANDLE);
		vkDestroyImageView(device, imageView, nullptr);
		vmaDestroyImage(allocator, image, allocation);
	}
}

void vk::ImageTransition(VkCommandBuffer cmd, VkImage image, VkFormat format, VkImageLayout currentLayout, VkImageLayout newLayout, VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask,
	VkPipelineStageFlagBits srcStageMask, VkPipelineStageFlagBits dstStageMask)
{
	VkImageMemoryBarrier barrier = {};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.oldLayout = currentLayout;
	barrier.newLayout = newLayout;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = image;
	barrier.subresourceRange.aspectMask = format != VK_FORMAT_D32_SFLOAT ? VK_IMAGE_ASPECT_COLOR_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
	barrier.subresourceRange.baseMipLevel = 0;
	barrier.subresourceRange.levelCount = 1;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount = 1;
	barrier.srcAccessMask = srcAccessMask; // No source access required // 0
	barrier.dstAccessMask = dstAccessMask; // VK_ACCESS_TRANSFER_WRITE_BIT

	vkCmdPipelineBarrier(
		cmd,
		srcStageMask, // VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
		dstStageMask, // VK_PIPELINE_STAGE_TRANSFER_BIT
		0,
		0, nullptr,
		0, nullptr,
		1, &barrier
	);
}

uint32_t vk::ComputeMipLevels(uint32_t width, uint32_t height)
{
	return static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1;
}

vk::Image vk::LoadTextureFromDisk(const std::string& path, Context& context, VkFormat format)
{

	int width, height, texChannels;
	//stbi_set_flip_vertically_on_load(1);
	stbi_uc* pixels = stbi_load(path.c_str(), &width, &height, &texChannels, 4);

	const auto imageSize = width * height * 4; // width * height * rgba

	if (!pixels)
	{
		ERROR("Failed to load texture: " + path);
	}

	// Create a buffer to which we can copy data to from CPU -> staging buffer
	vk::Buffer stagingBuffer = vk::CreateBuffer("stagingBuffer", context, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

	stagingBuffer.WriteToBuffer(pixels, imageSize);

	stbi_image_free(pixels);

	uint32_t mipLevels = ComputeMipLevels(width, height);

	vk::Image img = vk::CreateImageTexture2D(path, context, width, height, format, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);

	ExecuteSingleTimeCommands(context, [&](VkCommandBuffer cmd)
		{
			// Transition from LAYOUT_UNDEFINED to LAYOUT_TRANSFER_DST_OPTIMAL to copy contents
			// from buffer to the image
			ImageBarrier(
				cmd,
				img.image,
				0,
				VK_ACCESS_TRANSFER_WRITE_BIT,
				VK_IMAGE_LAYOUT_UNDEFINED,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, mipLevels, 0, 1 });

			VkBufferImageCopy bufferCopy = {
				.bufferOffset = 0,
				.bufferRowLength = 0,
				.bufferImageHeight = 0,
				.imageSubresource = VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
				.imageOffset = VkOffset3D{0,0,0},
				.imageExtent = VkExtent3D{ (uint32_t)width, (uint32_t)height, 1}
			};

			vkCmdCopyBufferToImage(cmd, stagingBuffer.buffer, img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bufferCopy);

			// Transition from DST_OPTIMAL Layout to SRC_OPTIMAL since it'll
			// be used as a SOURCE of a transfer operation during mip generation
			ImageBarrier(
				cmd,
				img.image,
				VK_ACCESS_TRANSFER_WRITE_BIT,
				VK_ACCESS_TRANSFER_READ_BIT,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });


			// now we need to process each mip to generate the mip maps
			for (uint32_t level = 1; level < mipLevels; level++)
			{
				VkImageBlit blit = {};
				blit.srcSubresource = VkImageSubresourceLayers{ VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 0, 1 };
				blit.srcOffsets[0] = { 0, 0, 0 };
				blit.srcOffsets[1] = { int32_t(width), int32_t(height), 1 };

				width >>= 1;
				if (width == 0) width = 1;
				height >>= 1;
				if (height == 0) height = 1;

				blit.dstSubresource = VkImageSubresourceLayers{
					VK_IMAGE_ASPECT_COLOR_BIT,
					level,
					0,
					1
				};

				blit.dstOffsets[0] = { 0,0,0 };
				blit.dstOffsets[1] = { int32_t(width), int32_t(height), 1 };

				vkCmdBlitImage(
					cmd,
					img.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
					img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
					1,
					&blit,
					VK_FILTER_LINEAR);


				ImageBarrier(cmd,
					img.image,
					VK_ACCESS_TRANSFER_WRITE_BIT,
					VK_ACCESS_TRANSFER_READ_BIT,
					VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
					VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
					VK_PIPELINE_STAGE_TRANSFER_BIT,
					VK_PIPELINE_STAGE_TRANSFER_BIT,
					VkImageSubresourceRange{
						VK_IMAGE_ASPECT_COLOR_BIT,
						level,
						1,
						0,
						1
					});
			}

			ImageBarrier(
				cmd,
				img.image,
				VK_ACCESS_TRANSFER_READ_BIT,
				VK_ACCESS_SHADER_READ_BIT,
				VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
				VkImageSubresourceRange{
					VK_IMAGE_ASPECT_COLOR_BIT,
					0,
					mipLevels,
					0,
					1
				}
			);

		});

	stagingBuffer.Destroy(context.device);

	return img;
}

vk::Image vk::CreateImageTexture2D(const std::string name, Context& context, uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags imageaspectFlags, uint32_t mipLevels)
{
	VkImageCreateInfo imageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.extent.width = width;
	imageInfo.extent.height = height;
	imageInfo.extent.depth = 1;
	imageInfo.mipLevels = mipLevels;
	imageInfo.arrayLayers = 1;
	imageInfo.format = format;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.usage = usage;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	VmaAllocationCreateInfo allocInfo{};
	allocInfo.flags = 0;
	allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

	VkImage image = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;

	VK_CHECK(vmaCreateImage(context.allocator, &imageInfo, &allocInfo, &image, &allocation, nullptr), "Failed to allocate memory for image");

	context.SetObjectName(context.device, (uint64_t)image, VK_OBJECT_TYPE_IMAGE, name.c_str());

	vmaSetAllocationName(context.allocator, allocation, name.c_str());

	// Now create the image view
	VkImageViewCreateInfo viewInfo{};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = image;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = format;
	viewInfo.components = VkComponentMapping{};
	viewInfo.subresourceRange = VkImageSubresourceRange{ imageaspectFlags, 0, VK_REMAINING_MIP_LEVELS, 0, 1 };

	VkImageView imageView = VK_NULL_HANDLE;
	VK_CHECK(vkCreateImageView(context.device, &viewInfo, nullptr, &imageView), "Failed to create image view");

	context.SetObjectName(context.device, (uint64_t)imageView, VK_OBJECT_TYPE_IMAGE_VIEW, name.c_str());


	return vk::Image(name, context.allocator, image, imageView, allocation);
}
