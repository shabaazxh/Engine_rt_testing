#pragma once
#include <volk/volk.h>
#include <vector>
#include <cassert>
#include <stdexcept>
#include <optional>
#include <fstream>
#include <utility>
#include "Utils.hpp"
#include "GLTF.hpp"
/*
    Pipeline abstraction which allows simpler and easier construction of pipelines
    Improvements:
        - Hot reloading shaders
*/

namespace vk {

    enum class PipelineType {
        GRAPHICS,
        COMPUTE,
        RAY
    };

    enum class ShaderType
    {
        VERTEX,
        FRAGMENT,
        GEOM,
        COMPUTE,
        HIT,
        MISS,
        RAYGEN
    };

    enum class VertexBinding
    {
        BIND,
        NONE
    };

    namespace
    {
        VkShaderStageFlagBits ShaderTypeToVkShaderStage(ShaderType type) {
            switch (type) {
            case ShaderType::VERTEX:
                return VK_SHADER_STAGE_VERTEX_BIT;
            case ShaderType::FRAGMENT:
                return VK_SHADER_STAGE_FRAGMENT_BIT;
            case ShaderType::GEOM:
                return VK_SHADER_STAGE_GEOMETRY_BIT;
            case ShaderType::COMPUTE:
                return VK_SHADER_STAGE_COMPUTE_BIT;
            case ShaderType::RAYGEN:
                return VK_SHADER_STAGE_RAYGEN_BIT_KHR;
            case ShaderType::HIT:
                return VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            case ShaderType::MISS:
                return VK_SHADER_STAGE_MISS_BIT_KHR;
            default:
                return VK_SHADER_STAGE_FLAG_BITS_MAX_ENUM;
            }
        }
    }

    namespace vk
    {
        class PipelineBuilder {
        public:
            PipelineBuilder(Context& context, PipelineType type, VertexBinding binding, uint32_t subpass)
                : context{ context }, subpass{ subpass }, pipelineType(type), binding{ binding }

            {}

            // Add a shader to the pipeline
            PipelineBuilder& AddShader(const std::string& shaderPath, ShaderType type) {
                shaders.push_back({ ShaderTypeToVkShaderStage(type), CreateShaderModule(shaderPath) });
                return *this;
            }

            PipelineBuilder& SetInputAssembly(VkPrimitiveTopology topology)
            {
                m_inputAssembly = {};
                m_inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
                m_inputAssembly.topology = topology;
                m_inputAssembly.primitiveRestartEnable = VK_FALSE;

                return *this;
            }

            PipelineBuilder& SetDynamicState(const std::vector<VkDynamicState>& dynamicStates) {

                m_dynamicStateInfo = {};
                m_dynamicStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
                m_dynamicStateInfo.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
                m_dynamicStateInfo.pDynamicStates = dynamicStates.data();

                return *this;
            }

            PipelineBuilder& SetRasterizationState(VkPolygonMode polygonMode, VkCullModeFlags cullMode, VkFrontFace frontFace, VkBool32 depthBiasEnable = VK_FALSE, float depthBiasConstantFactor = 0.0f)
            {
                m_rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
                m_rasterInfo.depthClampEnable = VK_FALSE;
                m_rasterInfo.rasterizerDiscardEnable = VK_FALSE;
                m_rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
                m_rasterInfo.cullMode = cullMode;
                m_rasterInfo.frontFace = frontFace;
                m_rasterInfo.depthBiasClamp = 0.0f;
                m_rasterInfo.lineWidth = 1.0f;
                m_rasterInfo.depthBiasEnable = depthBiasEnable;
                m_rasterInfo.depthBiasConstantFactor = depthBiasConstantFactor;

                return *this;
            }

            // Set the pipeline layout
            PipelineBuilder& SetPipelineLayout(const std::vector<VkDescriptorSetLayout>& layouts, const std::optional<VkPushConstantRange>& pushConstant = std::nullopt) {
                descriptorLayouts = layouts;

                m_pipelineLayout = {};
                m_pipelineLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
                m_pipelineLayout.setLayoutCount = static_cast<uint32_t>(descriptorLayouts.size());
                m_pipelineLayout.pSetLayouts = descriptorLayouts.data();

                if (pushConstant.has_value()) {
                    m_pipelineLayout.pushConstantRangeCount = 1;
                    m_pipelineLayout.pPushConstantRanges = &pushConstant.value();
                }
                else {
                    m_pipelineLayout.pushConstantRangeCount = 0;
                    m_pipelineLayout.pPushConstantRanges = nullptr;
                }
                

                return *this;
            }

            PipelineBuilder& SetSampling(VkSampleCountFlagBits sampleCount)
            {
                m_samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
                m_samplingInfo.rasterizationSamples = sampleCount;

                return *this;
            }

            PipelineBuilder& AddBlendAttachmentState
            (VkBool32 enableBlending = VK_FALSE,
                VkBlendFactor srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
                VkBlendFactor dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
                VkBlendOp colorBlendOp = VK_BLEND_OP_ADD,
                VkBlendFactor srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
                VkBlendFactor dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
                VkBlendOp alphaBlendOp = VK_BLEND_OP_ADD,
                VkColorComponentFlags colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT)
            {
                VkPipelineColorBlendAttachmentState blendAttState = {};
                blendAttState.blendEnable = enableBlending;
                blendAttState.srcColorBlendFactor = srcColorBlendFactor;
                blendAttState.dstColorBlendFactor = dstColorBlendFactor;
                blendAttState.colorBlendOp = colorBlendOp;
                blendAttState.srcAlphaBlendFactor = srcAlphaBlendFactor;
                blendAttState.dstAlphaBlendFactor = dstAlphaBlendFactor;
                blendAttState.alphaBlendOp = alphaBlendOp;
                blendAttState.colorWriteMask = colorWriteMask;

                m_blendAttachments.push_back(blendAttState);

                return *this;
            }

            PipelineBuilder& SetDepthState(VkBool32 depthTestEnable, VkBool32 depthWriteEnable, VkCompareOp depthCompareOp)
            {
                m_depthState = {};
                m_depthState.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
                m_depthState.depthTestEnable = depthTestEnable;
                m_depthState.depthWriteEnable = depthWriteEnable;
                m_depthState.depthCompareOp = depthCompareOp;
                m_depthState.minDepthBounds = 0.0f;
                m_depthState.maxDepthBounds = 1.0f;

                return *this;
            }

            // Set the render pass (only for graphics pipelines)
            PipelineBuilder& SetRenderPass(VkRenderPass renderPass) {
                this->renderPass = renderPass;

                return *this;
            }

            // Create the pipeline based on type (graphics or compute)
            std::pair<VkPipeline, VkPipelineLayout> Build() {

                if (pipelineType == PipelineType::GRAPHICS) {
                    return { CreateGraphicsPipeline(), pipelineLayout };
                }

                else if (pipelineType == PipelineType::COMPUTE) {
                    return { CreateComputePipeline(), pipelineLayout };
                }
                else
                {
                    std::cout << "Using RT pipeline" << std::endl;
                    return { CreateRayTracingPipeline(), pipelineLayout };
                }
            }

        private:
            Context& context;
            uint32_t subpass;
            PipelineType pipelineType;
            VkRenderPass renderPass = VK_NULL_HANDLE;
            VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
            std::vector<std::pair<VkShaderStageFlagBits, VkShaderModule>> shaders;
            std::vector<VkDescriptorSetLayout> descriptorLayouts;
            VkPushConstantRange pushConstantRange;

            VkPipelineInputAssemblyStateCreateInfo m_inputAssembly{};
            VkPipelineDynamicStateCreateInfo m_dynamicStateInfo{};
            VkPipelineViewportStateCreateInfo m_viewportInfo{};
            VkPipelineMultisampleStateCreateInfo m_samplingInfo{};
            std::vector<VkPipelineColorBlendAttachmentState> m_blendAttachments;
            VkPipelineDepthStencilStateCreateInfo m_depthState{};
            VkPipelineLayoutCreateInfo m_pipelineLayout{};
            VkPipelineRasterizationStateCreateInfo m_rasterInfo{};

            VertexBinding binding;

            std::vector<VkRayTracingShaderGroupCreateInfoKHR> shaderGroups{};
            VkShaderModule CreateShaderModule(const std::string& shaderPath) {

                std::vector<char> shaderCode = ReadShader(shaderPath);

                VkShaderModuleCreateInfo createInfo = {};
                createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
                createInfo.codeSize = shaderCode.size();
                createInfo.pCode = reinterpret_cast<const uint32_t*>(shaderCode.data());

                VkShaderModule shaderModule;
                if (vkCreateShaderModule(context.device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
                    throw std::runtime_error("Failed to create shader module!");
                }

                return shaderModule;
            }

            uint32_t aligned_size(uint32_t value, uint32_t alignment)
            {
                return (value + alignment - 1) & ~(alignment - 1);
            }

            VkPipeline CreateRayTracingPipeline()
            {
                std::vector<VkPipelineShaderStageCreateInfo> shaderStages;

                for (const auto& shader : shaders) {
                    VkPipelineShaderStageCreateInfo shaderStageInfo{};
                    shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                    shaderStageInfo.stage = shader.first;
                    shaderStageInfo.module = shader.second;
                    shaderStageInfo.pName = "main";
                    shaderStages.push_back(shaderStageInfo);
                }

                // Ray gen groups 
                VkRayTracingShaderGroupCreateInfoKHR rayGenGroup = {};
                rayGenGroup.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
                rayGenGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
                rayGenGroup.generalShader = 0;
                rayGenGroup.closestHitShader = VK_SHADER_UNUSED_KHR;
                rayGenGroup.anyHitShader = VK_SHADER_UNUSED_KHR;
                rayGenGroup.intersectionShader = VK_SHADER_UNUSED_KHR;
                shaderGroups.push_back(rayGenGroup);

                // Ray miss group
                VkRayTracingShaderGroupCreateInfoKHR missGroup = {};
                missGroup.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
                missGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
                missGroup.generalShader = 1;
                missGroup.closestHitShader = VK_SHADER_UNUSED_KHR;
                missGroup.anyHitShader = VK_SHADER_UNUSED_KHR;
                missGroup.intersectionShader = VK_SHADER_UNUSED_KHR;
                shaderGroups.push_back(missGroup);

                // Ray closes hit group 
                VkRayTracingShaderGroupCreateInfoKHR chGroup = {};
                chGroup.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
                chGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
                chGroup.generalShader = VK_SHADER_UNUSED_KHR;
                chGroup.closestHitShader = 2;
                chGroup.anyHitShader = VK_SHADER_UNUSED_KHR;
                chGroup.intersectionShader = VK_SHADER_UNUSED_KHR;
                shaderGroups.push_back(chGroup);

                VkResult res = vkCreatePipelineLayout(context.device, &m_pipelineLayout, nullptr, &pipelineLayout);

                VkRayTracingPipelineCreateInfoKHR rtPipeline = {};
                rtPipeline.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
                rtPipeline.stageCount = static_cast<uint32_t>(shaderStages.size());
                rtPipeline.pStages = shaderStages.data();
                rtPipeline.groupCount = static_cast<uint32_t>(shaderGroups.size());
                rtPipeline.pGroups = shaderGroups.data();
                rtPipeline.maxPipelineRayRecursionDepth = 1;
                rtPipeline.layout = pipelineLayout;

                PFN_vkCreateRayTracingPipelinesKHR vkCreateRayTracingPipelinesKHR =
                    (PFN_vkCreateRayTracingPipelinesKHR)vkGetDeviceProcAddr(context.device, "vkCreateRayTracingPipelinesKHR");

                if (!vkCreateRayTracingPipelinesKHR) {
                    std::cerr << "Failed to load vkCreateRayTracingPipelinesKHR!" << std::endl;
                }
                
                VkPipeline pipeline;
                    
                VK_CHECK(
                    vkCreateRayTracingPipelinesKHR(
                        context.device,
                        VK_NULL_HANDLE,
                        VK_NULL_HANDLE,
                        1,
                        &rtPipeline, 
                        nullptr, 
                        &pipeline
                    )
                , "Failed to create ray tracing pipeline.");


                for (auto& pair : shaders)
                    vkDestroyShaderModule(context.device, pair.second, nullptr);

                return pipeline;
            }

            // Create a graphics pipeline
            VkPipeline CreateGraphicsPipeline() {

                std::vector<VkPipelineShaderStageCreateInfo> shaderStages;

                for (const auto& shader : shaders) {
                    VkPipelineShaderStageCreateInfo shaderStageInfo{};
                    shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                    shaderStageInfo.stage = shader.first;
                    shaderStageInfo.module = shader.second;
                    shaderStageInfo.pName = "main";
                    shaderStages.push_back(shaderStageInfo);
                }

                VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
                vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        
                auto bindingDescription = Vertex::GetBindingDescription();
                auto attributeDescription = Vertex::GetAttributeDescriptions();

                if (binding == VertexBinding::BIND)
                {
                    vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(1);
                    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescription.size());
                    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
                    vertexInputInfo.pVertexAttributeDescriptions = attributeDescription.data();
                }
                else
                {
                    vertexInputInfo.vertexBindingDescriptionCount = 0; 
                    vertexInputInfo.vertexAttributeDescriptionCount = 0; 
                    vertexInputInfo.pVertexAttributeDescriptions = nullptr;
                    vertexInputInfo.pVertexBindingDescriptions = nullptr;
                }

                m_viewportInfo = {};
                m_viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
                m_viewportInfo.viewportCount = 1;
                m_viewportInfo.scissorCount = 1;

                VkPipelineColorBlendStateCreateInfo blendInfo = {};
                blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
                blendInfo.attachmentCount = static_cast<uint32_t>(m_blendAttachments.size());
                blendInfo.pAttachments = m_blendAttachments.data();

                VK_CHECK(vkCreatePipelineLayout(context.device, &m_pipelineLayout, nullptr, &pipelineLayout), "Failed to create pipeline layout");

                // Create graphics pipeline
                VkGraphicsPipelineCreateInfo pipelineInfo{};
                pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
                pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
                pipelineInfo.pStages = shaderStages.data();
                pipelineInfo.pVertexInputState = &vertexInputInfo;
                pipelineInfo.pInputAssemblyState = &m_inputAssembly;
                pipelineInfo.pTessellationState = nullptr;
                pipelineInfo.pViewportState = &m_viewportInfo;
                pipelineInfo.pRasterizationState = &m_rasterInfo;
                pipelineInfo.pMultisampleState = &m_samplingInfo;
                pipelineInfo.pDynamicState = &m_dynamicStateInfo;
                pipelineInfo.pDepthStencilState = &m_depthState;
                pipelineInfo.pColorBlendState = &blendInfo;
                pipelineInfo.layout = pipelineLayout;
                pipelineInfo.renderPass = renderPass;
                pipelineInfo.subpass = subpass;

                VkPipeline pipeline;
                if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
                    throw std::runtime_error("Failed to create graphics pipeline!");
                }

                for (auto& pair : shaders)
                    vkDestroyShaderModule(context.device, pair.second, nullptr);

                return pipeline;
            }

            // Create a compute pipeline
            VkPipeline CreateComputePipeline() {
                // Compute pipeline requires only a compute shader which should be the first shader provided
                
                VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
                computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
                computeShaderStageInfo.module = shaders[0].second;
                computeShaderStageInfo.pName = "main";

                VK_CHECK(vkCreatePipelineLayout(context.device, &m_pipelineLayout, nullptr, &pipelineLayout), "Failed to create compute pipeline layout");

                VkComputePipelineCreateInfo computePipelineInfo{};
                computePipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
                computePipelineInfo.stage = computeShaderStageInfo;
                computePipelineInfo.layout = pipelineLayout;

                VkPipeline pipeline;
                if (vkCreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &computePipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
                    throw std::runtime_error("Failed to create compute pipeline!");
                }

                for (auto& pair : shaders)
                    vkDestroyShaderModule(context.device, pair.second, nullptr);

                return pipeline;
            }

            // Reference: https://vulkan-tutorial.com/Drawing_a_triangle/Graphics_pipeline_basics/Shader_modules
            std::vector<char> ReadShader(const std::string& filename)
            {
                std::ifstream file(filename, std::ios::ate | std::ios::binary);
                if (!file.is_open())
                {
                    throw std::runtime_error("Failed to open shader file: " + filename);
                }

                size_t fileSize = (size_t)file.tellg();
                std::vector<char> buffer(fileSize);

                file.seekg(0);
                file.read(buffer.data(), fileSize);

                file.close();

                return buffer;
            }
        };
    }
}

