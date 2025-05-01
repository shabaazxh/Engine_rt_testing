#pragma once
#include <GLFW/glfw3.h>
#include <volk/volk.h>
#include <vk_mem_alloc.h>
#include <vector>
#include "Image.hpp"

namespace vk
{
	class Context
	{
	public:
		Context();
		void Destroy();
		bool MakeContext(uint32_t width, uint32_t height);
		void CreateLogicalDevice();
		void CreateAllocator();
		void CreateSwapchain();
		void TeardownSwapchain();
		void RecreateSwapchain();

		void SetObjectName(VkDevice device, uint64_t objectHandle, VkObjectType objectType, const char* name);

		GLFWwindow* window;
		VkInstance instance;

		VkPhysicalDevice pDevice;
		VkDevice device;
		VkSurfaceKHR surface;

		uint32_t graphicsFamilyIndex;
		uint32_t presentFamilyIndex;

		VkQueue graphicsQueue;
		VkQueue presentQueue;

		VkDebugUtilsMessengerEXT debugMessenger;
		bool enableDebugUtil;

		uint32_t numIndices;
		VmaAllocator allocator;

		// Swapchain
		VkSwapchainKHR swapchain;
		VkSwapchainKHR oldSwapchain;
		std::vector<VkImage> swapchainImages;
		std::vector<VkImageView> swapchainImageViews;
		std::vector<VkFramebuffer> swapchainFramebuffers;
		VkRenderPass renderPass;
		VkFormat swapchainFormat;
		VkExtent2D extent{};
		VkPresentModeKHR presentMode;
		bool isSwapchainOutdated;
		VkCommandPool transientCommandPool;
		VkDescriptorPool descriptorPool;
		PFN_vkSetDebugUtilsObjectNameEXT vkSetDebugUtilsObjectNameEXT;

	private:
		// Create transient pool once to use for one-time submit command buffers
		void CreateTransientCommandPool();
		void CreateDescriptorPool();
	};
}