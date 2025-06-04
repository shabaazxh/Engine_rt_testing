#pragma once
#include <volk/volk.h>
#include <memory>
#include "Image.hpp"
#include <vector>

namespace vk
{
	class Context;
	class Camera;
	class Scene;
	class Temporal
	{
	public:
		explicit Temporal(Context& context, std::shared_ptr<Scene>& scene, std::shared_ptr<Camera>& camera, Image& initialCandidates, Image& MotionVectors);
		~Temporal();

		void Execute(VkCommandBuffer cmd);
		void Update();
		void Resize();

		void CopyImageToImage(const Image& currentSpatialTemporalReservoirImage);

		Image& GetRenderTarget() { return m_RenderTarget; }
	private:
		void CreatePipeline();
		void CreateShaderBindingTable();
		void BuildDescriptors();

		Context& context;
		std::shared_ptr<Scene> scene;
		std::shared_ptr<Camera> camera;
		Image m_RenderTarget;
		Image m_PreviousImage;
		Image& initialCandidates;
		Image& MotionVectors;

		VkPipeline m_Pipeline;
		VkPipelineLayout m_PipelineLayout;
		std::vector<VkDescriptorSet> m_descriptorSets;
		VkDescriptorSetLayout m_descriptorSetLayout;

		uint32_t m_width;
		uint32_t m_height;
		VkPhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingPipelineProperties{};
		std::unique_ptr<Buffer> RayGenShaderBindingTable;
		std::unique_ptr<Buffer> MissShaderBindingTable;
		std::unique_ptr<Buffer> HitShaderBindingTable;
		std::vector<Buffer> m_rtxSettingsUBO;
	};
}