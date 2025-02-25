#include <volk/volk.h>
#include <vector>
#include <functional>
#include <iostream>
#include <memory>

namespace vk
{
    class Context;
    class Scene;
    class Camera;
    namespace ImGuiRenderer
    {
        static std::vector<std::function<void()>> ImGuiComponents;
        static std::vector<void*> textureIDs;

        inline void AddImGuiComponent(std::function<void()> func) {
            if (func) { // Check if the function is valid
                ImGuiComponents.push_back(func);
                std::cout << "Added component. Total: "
                    << ImGuiComponents.size() << std::endl;
            }
            else {
                std::cout << "Failed to add: Component is null!" << std::endl;
            }
        }

        // Todo: When resizing, somehow clear the list and add the new re-sized versions?
        void AddTexture(VkSampler sampler, VkImageView imageView, VkImageLayout imageLayout);


        static std::vector<VkDescriptorPoolSize> ImGuiPoolSizes = {
            {VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
            {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
            {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
            {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}
        };

        void Initialize(const Context& context);
        void Shutdown(const Context& context);
        void Update(const std::shared_ptr<Scene>& scene, const std::shared_ptr<Camera>& camera);
        void Render(VkCommandBuffer cmd, const Context& context, uint32_t imageIndex);

        inline VkDescriptorPool imGuiDescriptorPool;
        inline VkRenderPass imGuiRenderPass;
    }
}