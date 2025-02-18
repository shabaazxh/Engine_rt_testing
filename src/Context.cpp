#include <volk/volk.h>
#include "Context.hpp"
#include "Utils.hpp"
#include "RenderPass.hpp"

#include <unordered_set>
#include <string>
#include <optional>
#include <cassert>
#include <algorithm>
#include <cstring>

// Instance + Device
namespace
{
	std::unordered_set<std::string> GetInstanceLayers();
	std::unordered_set<std::string> GetInstanceExtensions();

	// Debug
	VKAPI_ATTR VkBool32 VKAPI_CALL DebugUtilsCallback(VkDebugUtilsMessageSeverityFlagBitsEXT aSeverity, VkDebugUtilsMessageTypeFlagsEXT aType, VkDebugUtilsMessengerCallbackDataEXT const* aData, void*);

    float ScoreDevice(VkPhysicalDevice pDevice, VkInstance instance, VkSurfaceKHR surface);
    VkPhysicalDevice SelectPhysicalDevice(VkInstance instance, VkSurfaceKHR surface);
    std::pair<std::optional<uint32_t>, std::optional<uint32_t>> GetGraphicsQueueFamily(VkPhysicalDevice pDevice, VkInstance instance, VkSurfaceKHR surface);
}

// Swapchain
namespace
{
    std::vector<VkSurfaceFormatKHR> GetSurfaceFormats(VkPhysicalDevice pDevice, VkSurfaceKHR surface);
    std::unordered_set<VkPresentModeKHR> GetPresentModes(VkPhysicalDevice pDevice, VkSurfaceKHR surface);

    std::vector<VkImage> GetSwapchainImages(VkDevice device, VkSwapchainKHR swapchain);
    std::vector<VkImageView> CreateSwapchainImageViews(VkDevice device, VkFormat format, const std::vector<VkImage>& images);
    std::vector<VkFramebuffer> CreateSwapchainFramebuffers(VkDevice device, std::vector<VkImageView>& swapchainImageViews, VkRenderPass renderPass, VkExtent2D extent);
    VkRenderPass CreateSwapchainRenderPass(VkDevice device, VkFormat format);
}

namespace
{
    std::unordered_set<std::string> GetInstanceLayers()
    {
        uint32_t numLayers = 0;
        vkEnumerateInstanceLayerProperties(&numLayers, nullptr);

        std::vector<VkLayerProperties> layers(numLayers);
        vkEnumerateInstanceLayerProperties(&numLayers, layers.data());

        std::unordered_set<std::string> res;
        for (const auto& layer : layers)
            res.insert(layer.layerName);

        return res;
    }

    std::unordered_set<std::string> GetInstanceExtensions()
    {
        uint32_t numExtensions = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &numExtensions, nullptr);


        std::vector<VkExtensionProperties> extensions(numExtensions);
        vkEnumerateInstanceExtensionProperties(nullptr, &numExtensions, extensions.data());

        std::unordered_set<std::string> res;
        for (const auto& extension : extensions)
            res.insert(extension.extensionName);

        return res;
    }

    VKAPI_ATTR VkBool32 VKAPI_CALL DebugUtilsCallback(VkDebugUtilsMessageSeverityFlagBitsEXT aSeverity, VkDebugUtilsMessageTypeFlagsEXT aType, VkDebugUtilsMessengerCallbackDataEXT const* aData, void*)
    {
        std::cerr << "[Validation ERROR]: " << aData->pMessage << std::endl;

        return VK_FALSE;
    }

    float ScoreDevice(VkPhysicalDevice pDevice, VkInstance instance, VkSurfaceKHR surface)
    {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(pDevice, &props);

        const auto MAJOR = VK_API_VERSION_MAJOR(props.apiVersion);
        const auto MINOR = VK_API_VERSION_MINOR(props.apiVersion);

        if (MAJOR < 1 || (MAJOR == 1 && MINOR < 2))
        {
            return -1.0f;
        }

        float score = 0.0f;

        uint32_t numExtensions = 0;
        vkEnumerateDeviceExtensionProperties(pDevice, nullptr, &numExtensions, nullptr);

        std::vector<VkExtensionProperties> extensions;
        vkEnumerateDeviceExtensionProperties(pDevice, nullptr, &numExtensions, extensions.data());


        bool supportsSwapchainExtension = false;
        for (auto& extension : extensions)
        {
            if (strcmp(extension.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0)
            {
                supportsSwapchainExtension = true;
            }
        }

        auto result = GetGraphicsQueueFamily(pDevice, instance, surface);
        bool hasPresent = result.second.has_value();
        bool hasGraphics = result.first.has_value();

        VkPhysicalDeviceFeatures2 features = {};
        features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        vkGetPhysicalDeviceFeatures2(pDevice, &features);

        // Prefer DISCRETE over INTEGRATED
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && features.features.samplerAnisotropy && (hasPresent && hasGraphics) && supportsSwapchainExtension)
        {
            score += 500.0f;
        }
        else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU && features.features.samplerAnisotropy && (hasPresent && hasGraphics) && supportsSwapchainExtension)
        {
            score += 100.0f;
        }

        return score;
    }

    VkPhysicalDevice SelectPhysicalDevice(VkInstance instance, VkSurfaceKHR surface)
    {
        uint32_t numberDevices = 0;
        vkEnumeratePhysicalDevices(instance, &numberDevices, nullptr);

        std::vector<VkPhysicalDevice> devices(numberDevices);
        vkEnumeratePhysicalDevices(instance, &numberDevices, devices.data());

        float topScore = -1.0f;
        VkPhysicalDevice pDevice = VK_NULL_HANDLE;

        for (const auto& device : devices)
        {
            float score = ScoreDevice(device, instance, surface);
            if (score > topScore)
            {
                topScore = score;
                pDevice = device;
            }
        }

        VkPhysicalDeviceFeatures2 features = {};
        features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        vkGetPhysicalDeviceFeatures2(pDevice, &features);

        return pDevice;
    }

    std::pair<std::optional<uint32_t>, std::optional<uint32_t>> GetGraphicsQueueFamily(VkPhysicalDevice pDevice, VkInstance instance, VkSurfaceKHR surface)
    {
        std::pair<std::optional<uint32_t>, std::optional<uint32_t>> queueFamilies = {};

        uint32_t numQueues = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pDevice, &numQueues, nullptr);

        std::vector<VkQueueFamilyProperties> families(numQueues);
        vkGetPhysicalDeviceQueueFamilyProperties(pDevice, &numQueues, families.data());

        for (uint32_t i = 0; i < numQueues; i++)
        {
            const auto& family = families[i];

            // check for graphics and compute operations support for current queue
            if ((family.queueFlags & VK_QUEUE_GRAPHICS_BIT) && (family.queueFlags & VK_QUEUE_COMPUTE_BIT))
            {
                queueFamilies.first = i;
            }

            // check for present support as well
            VkBool32 presentSupport = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(pDevice, i, surface, &presentSupport);

            if (presentSupport)
            {
                queueFamilies.second = i;
            }

            // early exit if we found both already
            if (queueFamilies.first.has_value() && queueFamilies.second.has_value())
            {
                break;
            }
        }
        return queueFamilies;
    }
}

namespace
{
    std::vector<VkSurfaceFormatKHR> GetSurfaceFormats(VkPhysicalDevice pDevice, VkSurfaceKHR surface)
    {
        uint32_t numFormats = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(pDevice, surface, &numFormats, nullptr);

        std::vector<VkSurfaceFormatKHR> formats(numFormats);
        vkGetPhysicalDeviceSurfaceFormatsKHR(pDevice, surface, &numFormats, formats.data());

        // Ensure we actually have some formats
        assert(!formats.empty());

        return formats;
    }
    std::unordered_set<VkPresentModeKHR> GetPresentModes(VkPhysicalDevice pDevice, VkSurfaceKHR surface)
    {
        uint32_t numModes = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(pDevice, surface, &numModes, nullptr);

        std::vector<VkPresentModeKHR> presentModes(numModes);
        vkGetPhysicalDeviceSurfacePresentModesKHR(pDevice, surface, &numModes, presentModes.data());

        std::unordered_set<VkPresentModeKHR> presents;

        for (const auto& mode : presentModes)
        {
            presents.insert(mode);
        }

        return presents;
    }
    std::vector<VkImage> GetSwapchainImages(VkDevice device, VkSwapchainKHR swapchain)
    {
        uint32_t numImages = 0;
        vkGetSwapchainImagesKHR(device, swapchain, &numImages, nullptr);

        std::vector<VkImage> swapchainImages(numImages);
        vkGetSwapchainImagesKHR(device, swapchain, &numImages, swapchainImages.data());

        return swapchainImages;
    }

    std::vector<VkImageView> CreateSwapchainImageViews(VkDevice device, VkFormat format, const std::vector<VkImage>& images)
    {
        std::vector<VkImageView> swapchainImageViews;
        for (size_t i = 0; i < images.size(); i++)
        {
            VkImageViewCreateInfo viewInfo{};
            viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewInfo.image = images[i];
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = format;
            viewInfo.components = VkComponentMapping{
                VK_COMPONENT_SWIZZLE_IDENTITY,
                VK_COMPONENT_SWIZZLE_IDENTITY,
                VK_COMPONENT_SWIZZLE_IDENTITY,
                VK_COMPONENT_SWIZZLE_IDENTITY
            };

            viewInfo.subresourceRange = VkImageSubresourceRange{
                VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1
            };

            VkImageView imageView = VK_NULL_HANDLE;
            VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &imageView), "Failed to create swapchain image views");

            swapchainImageViews.emplace_back(imageView);
        }

        return swapchainImageViews;
    }

    std::vector<VkFramebuffer> CreateSwapchainFramebuffers(VkDevice device, std::vector<VkImageView>& swapchainImageViews, VkRenderPass renderPass, VkExtent2D extent)
    {
        std::vector<VkFramebuffer> framebuffers;
        for (const auto& imageView : swapchainImageViews)
        {
            std::vector<VkImageView> attachments = { imageView };

            VkFramebufferCreateInfo fb_info{};
            fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            fb_info.renderPass = renderPass;
            fb_info.attachmentCount = static_cast<uint32_t>(attachments.size());
            fb_info.pAttachments = attachments.data();
            fb_info.width = extent.width;
            fb_info.height = extent.height;
            fb_info.layers = 1;

            VkFramebuffer framebuffer = VK_NULL_HANDLE;
            VK_CHECK(vkCreateFramebuffer(device, &fb_info, nullptr, &framebuffer), "Failed to create swapchain framebuffers.");

            framebuffers.push_back(framebuffer);
        }

        return framebuffers;
    }

    VkRenderPass CreateSwapchainRenderPass(VkDevice device, VkFormat format)
    {
        vk::RenderPass builder(device, 1);

        VkRenderPass renderPass = builder
            .AddAttachment(format, VK_SAMPLE_COUNT_1_BIT, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
            .AddColorAttachmentRef(0, 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)

            // External -> 0 : Color : Wait for presentation pass to finish?
            .AddDependency(VK_SUBPASS_EXTERNAL, 0, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_ACCESS_NONE, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_DEPENDENCY_BY_REGION_BIT)

            // 0 -> External : Color :
            .AddDependency(0, VK_SUBPASS_EXTERNAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_ACCESS_NONE, VK_DEPENDENCY_BY_REGION_BIT)


            .Build();


        return renderPass;
    }
}


vk::Context::Context() :
	window(nullptr),
	instance(VK_NULL_HANDLE),
	pDevice(VK_NULL_HANDLE),
	device(VK_NULL_HANDLE),
	surface(VK_NULL_HANDLE),
	graphicsFamilyIndex(0),
	presentFamilyIndex(0),
	graphicsQueue(VK_NULL_HANDLE),
	presentQueue(VK_NULL_HANDLE),
	debugMessenger(VK_NULL_HANDLE),
	enableDebugUtil(false),
    numIndices(0),
    allocator(VK_NULL_HANDLE),
    swapchain(VK_NULL_HANDLE),
    oldSwapchain{VK_NULL_HANDLE},
    renderPass(VK_NULL_HANDLE),
    swapchainFormat(VK_FORMAT_UNDEFINED),
    extent{},
    presentMode(VK_PRESENT_MODE_FIFO_KHR),
    isSwapchainOutdated(false),
    transientCommandPool(VK_NULL_HANDLE),
    descriptorPool(VK_NULL_HANDLE),
    vkSetDebugUtilsObjectNameEXT(VK_NULL_HANDLE)
{

}

void vk::Context::Destroy()
{
    vkDeviceWaitIdle(device);

    swapchainImages.clear();

    for (const auto& framebuffer : swapchainFramebuffers)
    {
        vkDestroyFramebuffer(device, framebuffer, nullptr);
    }

    for (const auto& imageView : swapchainImageViews)
    {
        vkDestroyImageView(device, imageView, nullptr);
    }

    vkDestroyRenderPass(device, renderPass, nullptr);

    if (oldSwapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device, oldSwapchain, nullptr);
    }

    vkDestroySwapchainKHR(device, swapchain, nullptr);

    if (transientCommandPool != VK_NULL_HANDLE)
    {
        vkDestroyCommandPool(device, transientCommandPool, nullptr);
    }

    if (descriptorPool != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    }

    if (surface != VK_NULL_HANDLE)
    {
        vkDestroySurfaceKHR(instance, surface, nullptr);
    }

    if (allocator != VK_NULL_HANDLE)
    {
        vmaDestroyAllocator(allocator);
    }

    if (device != VK_NULL_HANDLE)
    {
        vkDestroyDevice(device, nullptr);
    }

    if (debugMessenger != VK_NULL_HANDLE)
    {
        vkDestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
    }

    if (instance != VK_NULL_HANDLE)
    {
        vkDestroyInstance(instance, nullptr);
    }


    glfwDestroyWindow(window);
    glfwTerminate();
}

void vk::Context::TeardownSwapchain()
{
    vkDeviceWaitIdle(device);

    if (oldSwapchain != VK_NULL_HANDLE)
    {
        vkDestroySwapchainKHR(device, oldSwapchain, nullptr);
    }

    oldSwapchain = swapchain;

    swapchainImages.clear();
    for (const auto& framebuffer : swapchainFramebuffers)
    {
        vkDestroyFramebuffer(device, framebuffer, nullptr);
    }

    for (const auto& imageView : swapchainImageViews)
    {
        vkDestroyImageView(device, imageView, nullptr);
    }

    vkDestroyRenderPass(device, renderPass, nullptr);
}

void vk::Context::RecreateSwapchain()
{
    TeardownSwapchain();

    try {
        CreateSwapchain();
    }
    catch (...)
    {
        throw;
    }

}

void vk::Context::SetObjectName(VkDevice device, uint64_t objectHandle, VkObjectType objectType, const char* name)
{
    if (vkSetDebugUtilsObjectNameEXT) {
        VkDebugUtilsObjectNameInfoEXT nameInfo = {};
        nameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
        nameInfo.objectType = objectType;
        nameInfo.objectHandle = objectHandle;
        nameInfo.pObjectName = name;
        vkSetDebugUtilsObjectNameEXT(device, &nameInfo);
    }
}

void vk::Context::CreateLogicalDevice()
{
    float queuePriorities[1] = { 1.0f };

    VkDeviceQueueCreateInfo queueInfo = {};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = graphicsFamilyIndex;
    queueInfo.pQueuePriorities = queuePriorities;
    queueInfo.queueCount = 1;

    std::vector<const char*> extensions{
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_RAY_QUERY_EXTENSION_NAME
    };

    VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
        .accelerationStructure = VK_TRUE
    };

    VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
        .pNext = &asFeatures,
        .rayQuery = VK_TRUE
    };

    VkPhysicalDeviceBufferDeviceAddressFeatures bufferFeatures = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
        .pNext = &rayQueryFeatures,
        .bufferDeviceAddress = VK_TRUE
    };

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
        .pNext = &bufferFeatures,
        .rayTracingPipeline = VK_TRUE
    };

    VkPhysicalDeviceScalarBlockLayoutFeaturesEXT scalarBlockFeatures
    {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES_EXT,
        .pNext = &rtPipelineFeatures,
        .scalarBlockLayout = VK_TRUE
    };

    VkPhysicalDeviceFeatures2 deviceFeatures2{};
    deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    deviceFeatures2.pNext = &scalarBlockFeatures;
    deviceFeatures2.features.samplerAnisotropy = VK_TRUE;

    VkDeviceCreateInfo deviceInfo = {};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    deviceInfo.ppEnabledExtensionNames = extensions.data();
    deviceInfo.pNext = &deviceFeatures2;

    VK_CHECK(vkCreateDevice(pDevice, &deviceInfo, nullptr, &device), "Failed to create logical device.");
}

void vk::Context::CreateAllocator()
{
    VkPhysicalDeviceProperties props = {};
    vkGetPhysicalDeviceProperties(pDevice, &props);

    VmaVulkanFunctions functions = {};
    functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    functions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
    functions.vkGetPhysicalDeviceProperties = vkGetPhysicalDeviceProperties;
    functions.vkGetPhysicalDeviceMemoryProperties = vkGetPhysicalDeviceMemoryProperties;

    VmaAllocatorCreateInfo allocInfo = {};
    allocInfo.vulkanApiVersion = props.apiVersion;
    allocInfo.physicalDevice = pDevice;
    allocInfo.device = device;
    allocInfo.instance = instance;
    allocInfo.pVulkanFunctions = &functions;
    allocInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

    VK_CHECK(vmaCreateAllocator(&allocInfo, &allocator), "Failed to create VmaAllocator.");
}

void vk::Context::CreateSwapchain()
{
    const auto formats = GetSurfaceFormats(pDevice, surface);
    const auto presentModes = GetPresentModes(pDevice, surface);

    VkSurfaceFormatKHR swapFormat = formats[0];
    for (const auto& format : formats)
    {
        if (format.format == VK_FORMAT_B8G8R8A8_UNORM && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            swapFormat = format;
            break;
        }


        if (format.format == VK_FORMAT_B8G8R8A8_UNORM && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            swapFormat = format;
            break;
        }

    }

    swapchainFormat = swapFormat.format;

    // Use FIFO_RELAXED if it's available
    presentMode = VK_PRESENT_MODE_FIFO_KHR;
    if (presentModes.count(VK_PRESENT_MODE_FIFO_RELAXED_KHR)) {
        presentMode = VK_PRESENT_MODE_FIFO_RELAXED_KHR;
    }

    VkSurfaceCapabilitiesKHR caps = {};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pDevice, surface, &caps);

    uint32_t imageCount = caps.minImageCount + 1;

    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount)
    {
        imageCount = caps.maxImageCount;
    }

    extent = caps.currentExtent;

    if (extent.width == std::numeric_limits<uint32_t>::max())
    {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        const auto& min = caps.minImageExtent;
        const auto& max = caps.maxImageExtent;

        extent.width = std::clamp(uint32_t(width), min.width, max.width);
        extent.height = std::clamp(uint32_t(height), min.height, max.height);
    }

    // Swapchain create info
    VkSwapchainCreateInfoKHR swapchainCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = surface,
        .minImageCount = imageCount,
        .imageFormat = swapFormat.format,
        .imageColorSpace = swapFormat.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .preTransform = caps.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = presentMode,
        .clipped = VK_TRUE,
        .oldSwapchain = oldSwapchain
    };

    // Set max frames in flight
    vk::MAX_FRAMES_IN_FLIGHT = imageCount;

    if (numIndices <= 1)
    {
        swapchainCreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }
    else
    {
        uint32_t ind[] = { graphicsFamilyIndex, presentFamilyIndex };
        swapchainCreateInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        swapchainCreateInfo.queueFamilyIndexCount = numIndices;
        swapchainCreateInfo.pQueueFamilyIndices = ind;
    }

    VK_CHECK(vkCreateSwapchainKHR(device, &swapchainCreateInfo, nullptr, &swapchain), "Failed to create swapchain");

    renderPass = CreateSwapchainRenderPass(device, swapchainFormat);
    swapchainImages = GetSwapchainImages(device, swapchain);
    swapchainImageViews = CreateSwapchainImageViews(device, swapchainFormat, swapchainImages);
    swapchainFramebuffers = CreateSwapchainFramebuffers(device, swapchainImageViews, renderPass, extent);
}

bool vk::Context::MakeContext(uint32_t width, uint32_t height)
{
    if (volkInitialize() != VK_SUCCESS) {
        ERROR("Failed to initialize Volk.");
        return false;
    }

    const std::unordered_set<std::string> supportedLayers = GetInstanceLayers();
    const std::unordered_set<std::string> supportedExtensions = GetInstanceExtensions();

    std::vector<const char*> enabledLayers;
    std::vector<const char*> enabledExtensions;

#if !defined(NDEBUG)
    if (supportedLayers.count("VK_LAYER_KHRONOS_validation"))
    {
        enabledLayers.emplace_back("VK_LAYER_KHRONOS_validation");
    }

    if (supportedExtensions.count("VK_EXT_debug_utils"))
    {
        enableDebugUtil = true;
        enabledExtensions.emplace_back("VK_EXT_debug_utils");
    }
#endif

    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(width, height, "Vulkan", nullptr, nullptr);

    if (!window)
    {
        std::cerr << "Failed to create GLFW window" << std::endl;
    }

    uint32_t reqExtCount = 0;
    const char** requiredExt = glfwGetRequiredInstanceExtensions(&reqExtCount);

    for (uint32_t i = 0; i < reqExtCount; i++)
    {
        if (!supportedExtensions.count(requiredExt[i]))
        {
            std::runtime_error("glfw/vulkan required extension is not supported");
        }

        enabledExtensions.emplace_back(requiredExt[i]);
    }

    // Output the enabled layers and extensions
    //for (const auto& layer : enabledLayers)
    //    std::fprintf(stderr, "Enabled layer: %s\n", layer);
    //for (const auto& extension : enabledExtensions)
    //    std::fprintf(stderr, "Enabled inst extension: %s\n", extension);

    // Prepare debugger messenger info
    VkDebugUtilsMessengerCreateInfoEXT debugInfo = {};
    if (enableDebugUtil)
    {
        debugInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debugInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debugInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debugInfo.pfnUserCallback = &DebugUtilsCallback;
        debugInfo.pUserData = nullptr;
    }

    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "vkApp";
    appInfo.applicationVersion = 2000;
    appInfo.apiVersion = VK_MAKE_API_VERSION(0, 1, 3, 0);

    VkInstanceCreateInfo instanceInfo = {};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.enabledLayerCount = uint32_t(enabledLayers.size());
    instanceInfo.ppEnabledLayerNames = enabledLayers.data();
    instanceInfo.enabledExtensionCount = uint32_t(enabledExtensions.size());
    instanceInfo.ppEnabledExtensionNames = enabledExtensions.data();
    instanceInfo.pApplicationInfo = &appInfo;

    if (enableDebugUtil)
    {
        debugInfo.pNext = instanceInfo.pNext;
        instanceInfo.pNext = &debugInfo;
    }

    VkResult res = vkCreateInstance(&instanceInfo, nullptr, &instance);

    if (res != VK_SUCCESS)
    {
        return false;
    }

    volkLoadInstance(instance);

    // Create logical device (graphics family first)
    // create window surface
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface))
    {
        throw std::runtime_error("Failed to create a GLFW window surface.");
    }


    // Select the physical device
    pDevice = SelectPhysicalDevice(instance, surface);

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pDevice, &props);
    std::fprintf(stderr, "Device: %s\n", props.deviceName);

    std::pair<std::optional<uint32_t>, std::optional<uint32_t>> qFamilies = GetGraphicsQueueFamily(pDevice, instance, surface);

    graphicsFamilyIndex = qFamilies.first.has_value() ? qFamilies.first.value() : 0;
    presentFamilyIndex = qFamilies.second.has_value() ? qFamilies.second.value() : 0;

    numIndices = graphicsFamilyIndex != presentFamilyIndex ? 2 : 1;

    CreateLogicalDevice();

    if (device == VK_NULL_HANDLE)
    {
        std::runtime_error("Failed to create logical device.");
    }

    vkGetDeviceQueue(device, graphicsFamilyIndex, 0, &graphicsQueue);
    vkGetDeviceQueue(device, presentFamilyIndex, 0, &presentQueue);

    CreateAllocator();
    CreateTransientCommandPool();
    CreateDescriptorPool();

    assert(graphicsQueue != VK_NULL_HANDLE);
    assert(presentQueue != VK_NULL_HANDLE);

    vkSetDebugUtilsObjectNameEXT = (PFN_vkSetDebugUtilsObjectNameEXT)vkGetInstanceProcAddr(instance, "vkSetDebugUtilsObjectNameEXT");

    CreateSwapchain();

    // Set max anisotropic level
    maxAnisotropic = props.limits.maxSamplerAnisotropy;

    return true;
}

void vk::Context::CreateTransientCommandPool()
{
    VkCommandPoolCreateInfo cmdPoolInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = graphicsFamilyIndex
    };

    VK_CHECK(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &transientCommandPool), "Failed to create transient command pool.");
}

void vk::Context::CreateDescriptorPool()
{
    VkDescriptorPoolSize bufferPoolSize{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER };
    bufferPoolSize.descriptorCount = 512;
    VkDescriptorPoolSize samplerPoolSize{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER };
    samplerPoolSize.descriptorCount = 512;
    VkDescriptorPoolSize storagePoolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
    storagePoolSize.descriptorCount = 512;

    std::vector<VkDescriptorPoolSize> poolSize = { bufferPoolSize, samplerPoolSize, storagePoolSize };

    VkDescriptorPoolCreateInfo info{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    info.poolSizeCount = static_cast<uint32_t>(poolSize.size());
    info.pPoolSizes = poolSize.data();
    info.maxSets = 1536;

    VK_CHECK(vkCreateDescriptorPool(device, &info, nullptr, &descriptorPool), "Failed to create descriptor pool");
}