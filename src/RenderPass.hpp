#pragma once
#include <volk/volk.h>
#include <vector>
#include <cassert>
#include <stdexcept>
#include <optional>
#include <string>

namespace vk {
    class RenderPass {
    public:
        RenderPass(VkDevice device, uint32_t numSubpasses) : m_Device(device), m_NumSubpasses(numSubpasses) {
            m_ColorAttachmentRefs.resize(1);
            m_InputAttachmentRefs.resize(1);
            m_DepthAttachmentRefs.resize(1);
        }

        // Add an attachment to the render pass
        RenderPass& AddAttachment(VkFormat format, VkSampleCountFlagBits samples,
            VkAttachmentLoadOp loadOp, VkAttachmentStoreOp storeOp,
            VkImageLayout initialLayout, VkImageLayout finalLayout) {

            VkAttachmentDescription attachment{};
            attachment.format = format;
            attachment.samples = samples;
            attachment.loadOp = loadOp;
            attachment.storeOp = storeOp;
            attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachment.initialLayout = initialLayout;
            attachment.finalLayout = finalLayout;

            m_Attachments.push_back(attachment);
            return *this;
        }

        // Add a color attachment reference for a subpass
        RenderPass& AddColorAttachmentRef(uint32_t subpassIndex, uint32_t attachmentIndex, VkImageLayout layout) {
            assert(subpassIndex < m_NumSubpasses);
            VkAttachmentReference ref{};
            ref.attachment = attachmentIndex;
            ref.layout = layout;
            m_ColorAttachmentRefs[subpassIndex].push_back(ref);
            return *this;
        }

        // Input attachment reference - Not used in current implementation
        RenderPass& AddInputAttachmentRef(uint32_t subpassIndex, uint32_t attachmentIndex, VkImageLayout layout) {
            assert(subpassIndex < m_NumSubpasses);
            VkAttachmentReference ref{};
            ref.attachment = attachmentIndex;
            ref.layout = layout;
            m_InputAttachmentRefs[subpassIndex].push_back(ref);
            return *this;
        }

        // Set the depth attachment and specify the subpass it's for 
        RenderPass& SetDepthAttachmentRef(uint32_t subpassIndex, uint32_t attachmentIndex, VkImageLayout layout) {
            assert(subpassIndex < m_NumSubpasses);
            VkAttachmentReference depthRef{};
            depthRef.attachment = attachmentIndex;
            depthRef.layout = layout;
            m_DepthAttachmentRefs[subpassIndex] = depthRef;
            return *this;
        }

        // Subpass dependency
        RenderPass& AddDependency(uint32_t srcSubpass, uint32_t dstSubpass,
            VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
            VkPipelineStageFlags dstStage, VkAccessFlags dstAccess,
            VkDependencyFlags dependencyFlags = 0) {

            VkSubpassDependency dependency{};
            dependency.srcSubpass = srcSubpass;
            dependency.dstSubpass = dstSubpass;
            dependency.srcStageMask = srcStage;
            dependency.srcAccessMask = srcAccess;
            dependency.dstStageMask = dstStage;
            dependency.dstAccessMask = dstAccess;
            dependency.dependencyFlags = dependencyFlags;
            m_Dependencies.push_back(dependency);
            return *this;
        }

        VkRenderPass Build() {
            for (uint32_t i = 0; i < m_NumSubpasses; ++i) {
                VkSubpassDescription subpass{};
                subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;

                // Color attachments
                subpass.colorAttachmentCount = static_cast<uint32_t>(m_ColorAttachmentRefs[i].size());
                subpass.pColorAttachments = m_ColorAttachmentRefs[i].data();

                // Input attachments
                subpass.inputAttachmentCount = static_cast<uint32_t>(m_InputAttachmentRefs[i].size());
                subpass.pInputAttachments = m_InputAttachmentRefs[i].data();

                // Depth attachment (optional)
                if (m_DepthAttachmentRefs[i].has_value()) {
                    subpass.pDepthStencilAttachment = &m_DepthAttachmentRefs[i].value();
                }
                else {
                    subpass.pDepthStencilAttachment = nullptr;
                }

                m_Subpasses.push_back(subpass);
            }

            // Create render pass info
            VkRenderPassCreateInfo renderPassInfo{ VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
            renderPassInfo.attachmentCount = static_cast<uint32_t>(m_Attachments.size());
            renderPassInfo.pAttachments = m_Attachments.data();
            renderPassInfo.subpassCount = static_cast<uint32_t>(m_Subpasses.size());
            renderPassInfo.pSubpasses = m_Subpasses.data();
            renderPassInfo.dependencyCount = static_cast<uint32_t>(m_Dependencies.size());
            renderPassInfo.pDependencies = m_Dependencies.data();

            VkRenderPass renderPass;
            if (vkCreateRenderPass(m_Device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create render pass");
            }

            return renderPass;
        }

    private:
        VkDevice m_Device;
        uint32_t m_NumSubpasses;

        // Attachments and subpasses
        std::vector<VkAttachmentDescription> m_Attachments;
        std::vector<std::vector<VkAttachmentReference>> m_ColorAttachmentRefs;
        std::vector<std::vector<VkAttachmentReference>> m_InputAttachmentRefs;
        std::vector<std::optional<VkAttachmentReference>> m_DepthAttachmentRefs;
        std::vector<VkSubpassDependency> m_Dependencies;
        std::vector<VkSubpassDescription> m_Subpasses;
    };
}
