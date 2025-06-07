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

	class TemporalCompute
	{
	public:
		explicit TemporalCompute(Context& context, std::shared_ptr<Scene>& scene, std::shared_ptr<Camera>& camera, Image& initial_candidates, Image& hit_world_positions, Image& hit_normals, Image& motion_vectors);
		~TemporalCompute();

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
		Image& initial_candidates;
		Image& motion_vectors;
		Image& hit_world_positions;
		Image& hit_normals;

		VkPipeline m_Pipeline;
		VkPipelineLayout m_PipelineLayout;
		std::vector<VkDescriptorSet> m_descriptorSets;
		VkDescriptorSetLayout m_descriptorSetLayout;

		uint32_t m_width;
		uint32_t m_height;

		std::vector<Buffer> m_uniformBuffers;
	};
}