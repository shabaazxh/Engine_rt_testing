#include "Context.hpp"
#include "Buffer.hpp"
#include "Utils.hpp"

vk::Buffer::Buffer() noexcept : buffer{ VK_NULL_HANDLE }, allocation{ VK_NULL_HANDLE }, allocator{ VK_NULL_HANDLE }, name{ "" } {}

vk::Buffer::Buffer(const std::string& name, VmaAllocator allocator, VkBuffer buffer, VmaAllocation allocation) :
    buffer{ buffer }, allocation{ allocation }, allocator{ allocator }, name{ name } {}

vk::Buffer::Buffer(Buffer&& other) noexcept : 
    buffer(std::exchange(other.buffer, VK_NULL_HANDLE)),
    allocation(std::exchange(other.allocation, VK_NULL_HANDLE)),
    allocator(std::exchange(other.allocator, VK_NULL_HANDLE)),
    name(std::exchange(other.name, "")) {}

vk::Buffer& vk::Buffer::operator=(vk::Buffer&& other) noexcept
{
    std::swap(buffer, other.buffer);
	std::swap(allocation, other.allocation);
	std::swap(allocator, other.allocator);
    std::swap(name, other.name);

	return *this;
}

void vk::Buffer::Destroy(VkDevice device)
{
    if (buffer != VK_NULL_HANDLE)
    {
        assert(allocator != VK_NULL_HANDLE);
        assert(allocation != VK_NULL_HANDLE);
        vmaDestroyBuffer(allocator, buffer, allocation);
    }
}

vk::Buffer vk::CreateBuffer(const std::string& name, Context& context, VkDeviceSize bSize, VkBufferUsageFlags usage, VmaAllocationCreateFlags memoryFlags, VmaMemoryUsage memUsage)
{
	VkBufferCreateInfo bufferInfo = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = bSize,
		.usage = usage
	};

	VmaAllocationCreateInfo allocInfo = {
		.flags = memoryFlags,
		.usage = memUsage
	};

	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;

	VK_CHECK(vmaCreateBuffer(context.allocator, &bufferInfo, &allocInfo, &buffer, &allocation, nullptr), "Failed to create & allocate buffer");

    vmaSetAllocationName(context.allocator, allocation, name.c_str());
    context.SetObjectName(context.device, (uint64_t)buffer, VK_OBJECT_TYPE_BUFFER, name.c_str());

	return Buffer(name, context.allocator, buffer, allocation);
}

void vk::CreateAndUploadBuffer(vk::Context& context, const void* data, VkDeviceSize size, VkBufferUsageFlags usage, vk::Buffer& destinationBuffer)
{
    // Create the destination buffer
    destinationBuffer = vk::CreateBuffer("buffer", context, size,
        usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, 0, VMA_MEMORY_USAGE_AUTO);

    // Create the staging buffer
    vk::Buffer stagingBuffer = vk::CreateBuffer("staging buffer", context, size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        VMA_MEMORY_USAGE_AUTO);

    // Write data to the staging buffer
    stagingBuffer.WriteToBuffer(data, size);

    // Perform the copy operation using single-time commands
    ExecuteSingleTimeCommands(context, [&](VkCommandBuffer cmd) {
        
        VkBufferCopy copy = {
            .size = size
        };
        vkCmdCopyBuffer(cmd, stagingBuffer.buffer, destinationBuffer.buffer, 1, &copy);

        VkAccessFlags dstAccessMask = (usage & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) ? VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT : VK_ACCESS_INDEX_READ_BIT;

        VkBufferMemoryBarrier bufferBarrier = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = dstAccessMask,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = destinationBuffer.buffer,
            .size = VK_WHOLE_SIZE,
        };

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 0, nullptr, 1, &bufferBarrier, 0, nullptr);

        });

    stagingBuffer.Destroy(context.device);
}
