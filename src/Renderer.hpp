#pragma once

// Creates command pool, command buffers
#include <volk/volk.h>
#include <memory>
#include <vector>
#include "DepthPrepass.hpp"
#include "ForwardPass.hpp"
#include "PresentPass.hpp"
#include "Scene.hpp"
#include "Camera.hpp"
#include "ShadowMap.hpp"
#include "Composite.hpp"
#include "RayPass.hpp"
#include "Temporal.hpp"
#include "History.hpp"
#include "Spatial.hpp"
#include "MotionVectors.hpp"
#include "TemporalCompute.hpp"
#include "SpatialCompute.hpp"

namespace vk
{
	class Context;
	class Renderer
	{
	public:
		Renderer() = default;
		Renderer(Context& context);

		void Destroy();

		void Render(double deltaTime);
		void Update(double deltaTime);

		// Should be moved out of renderer when we do better input/controls
		static void glfwHandleKeyboard(GLFWwindow* window, int key, int scancode, int action, int mods);
		static void glfwMouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
		static void glfwCallbackMotion(GLFWwindow* window, double x, double y);

	private:
		void CreateResources();
		void CreateFences();
		void CreateSemaphores();
		void CreateCommandPool();
		void AllocateCommandBuffers();

		void Submit();
		void Present(uint32_t imageIndex);

	private:
		Context& context;
		std::vector<VkFence> m_Fences;
		std::vector<VkSemaphore> m_imageAvailableSemaphores;
		std::vector<VkSemaphore> m_renderFinishedSemaphores;
		std::vector<VkCommandBuffer> m_commandBuffers;
		std::vector<VkCommandPool> m_commandPool;

		std::shared_ptr<Scene> m_scene;

		std::unique_ptr<DepthPrepass>     m_DepthPrepass;
		std::unique_ptr<ForwardPass>	  m_ForwardPass;
		std::unique_ptr<ShadowMap>		  m_ShadowMap;
		std::unique_ptr<Composite>        m_CompositePass;
		std::unique_ptr<PresentPass>	  m_PresentPass;
		std::unique_ptr<RayPass>          m_RayPass;
		std::unique_ptr<MotionVectors>    m_MotionVectorsPass;
		std::unique_ptr<Temporal>         m_TemporalPass;
		std::unique_ptr<TemporalCompute>  m_TemporalComputePass;
		std::unique_ptr<SpatialCompute>   m_SpatialComputePass;
		std::unique_ptr<Spatial>          m_SpatialPass;
		std::unique_ptr<History>          m_HistoryPass;
		std::shared_ptr<Camera> m_camera;
		MaterialManager m_materialManager;
	};
}