#include "Context.hpp"
#include "Renderer.hpp"
#include "Utils.hpp"
#include "Light.hpp"

#include <glm/gtc/random.hpp>

namespace
{
	// This should be placed elsewhere. Put here for simplicity while testing 
	// Don't really need to define these, can pass the pos, dir, up directly to camera constructor
	// Camera default values 
	constexpr glm::vec3 cameraPos = glm::vec3(1.0f, 1.0f, 1.0f); //1.0f, 2.0f, -24.0f
	constexpr glm::vec3 cameraDir = glm::vec3(1.0f, 1.0f, -1.0f);
	constexpr glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0);
}

vk::Renderer::Renderer(Context& context) : context{context}
{
	std::printf("Launching Renderer\n");
	vk::renderType = RenderType::FORWARD;

	CreateResources();

	m_materialManager.materials.reserve(100);
	for (int i = 0; i < 100; ++i) {
		m_materialManager.materials.emplace_back(context);
	}

	m_materialManager.Setup(context);

	// Samplers
	repeatSamplerAniso	 	  = CreateSampler(context, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_TRUE,  VK_COMPARE_OP_LESS_OR_EQUAL);
	repeatSampler			  = CreateSampler(context, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_FALSE, VK_COMPARE_OP_LESS_OR_EQUAL);
	clampToEdgeSamplerAniso   = CreateSampler(context, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_FALSE, VK_COMPARE_OP_GREATER);
	
	// Camera
	m_camera = std::make_shared<Camera>(context, cameraPos, glm::normalize(cameraPos + cameraDir), up, context.extent.width / (float)context.extent.height);
	
	// GLFW callbacks
	glfwSetWindowUserPointer(context.window, m_camera.get());
	glfwSetKeyCallback(context.window, &glfwHandleKeyboard);
	glfwSetMouseButtonCallback(context.window, glfwMouseButtonCallback);
	glfwSetCursorPosCallback(context.window, glfwCallbackMotion);

	// Define Light sources
	Light directionalLight;
	directionalLight.Type = LightType::Directional;
	directionalLight.position = glm::vec4(-8.161, 23.6f, 4.0f, 1.0f); // -0.2972
	directionalLight.colour   = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);

	std::vector<glm::vec4> spotLightPositions;

	for (size_t i = 0; i < 25; i++)
	{
		spotLightPositions.push_back(glm::vec4(-9.0 + i * 0.8, 0.2f, 0.5f, 1.0f));
	}

	// Create the scene which will store models and lights
	// Add GLTF to the scene 
	// Add a directional light source defined earlier 
	// Sponza is huge ( physical size not disc space ) when loaded
	// so i reduced it significantly when rendering meshes (see DrawGLTF) in Scene.cpp
	auto gltf = vk::LoadGLTF(context, "assets/GLTF/Sponza/Sponza.gltf");

	m_scene = std::make_shared<Scene>(context, m_materialManager);

	m_scene->AddModel(gltf, m_materialManager);
	m_scene->AddLightSource(directionalLight);

	//for (auto& light : spotLightPositions)
	//{
	//	auto box = vk::LoadGLTF(context, "assets/GLTF/Box/BoxTextured.gltf");
	//	box.position = { light.x, light.y, light.z, 1.0 };
	//	m_scene->AddModel(box, m_materialManager);
	//}

	// Loop through the positions and instantiate a light 
	// and pass to the scene to add the lights to the scene
	for (const auto& position : spotLightPositions)
	{
		Light spotLight = {};
		spotLight.Type = LightType::Spot;
		spotLight.position = position;
		spotLight.colour = glm::vec4(
			glm::linearRand(0.0f, 1.0f),
			glm::linearRand(0.0f, 1.f),
			glm::linearRand(0.0f, 1.0f),
			1.0f 
		);		
		m_scene->AddLightSource(spotLight);
	}

	// Models should not all be loaded 
	// We have the data to build materials 
	m_materialManager.BuildMaterials(context);

	std::cout << "Number of Lights: " << m_scene->GetLights().size() << std::endl;

	// Renderer passes
	m_ShadowMap	    = std::make_unique<ShadowMap>(context, m_scene);
	m_DepthPrepass  = std::make_unique<DepthPrepass>(context, m_scene, m_camera);
	m_ForwardPass   = std::make_unique<ForwardPass>(context, m_ShadowMap->GetRenderTarget(), m_DepthPrepass->GetRenderTarget(), m_scene, m_camera);
	m_CompositePass = std::make_unique<Composite>(context, m_ForwardPass->GetRenderTarget(), m_ForwardPass->GetRenderTarget());
	m_PresentPass   = std::make_unique<PresentPass>(context, m_CompositePass->GetRenderTarget()); 
	m_RayPass       = std::make_unique<RayPass>(context, m_scene, m_camera);
}

void vk::Renderer::Destroy()
{
	vkDeviceWaitIdle(context.device);

	m_DepthPrepass.reset();
	m_ForwardPass.reset();
	m_ShadowMap.reset();
	m_CompositePass.reset();
	m_PresentPass.reset();
	m_camera.reset();
	m_scene->Destroy();

	vkDestroySampler(context.device, repeatSamplerAniso, nullptr);
	vkDestroySampler(context.device, repeatSampler, nullptr);
	vkDestroySampler(context.device, clampToEdgeSamplerAniso, nullptr);

	m_materialManager.Destroy(context);

	for (auto& fence : m_Fences)
	{
		vkDestroyFence(context.device, fence, nullptr);
	}

	for (auto& semaphore : m_imageAvailableSemaphores)
	{
		vkDestroySemaphore(context.device, semaphore, nullptr);
	}

	for (auto& semaphore : m_renderFinishedSemaphores)
	{
		vkDestroySemaphore(context.device, semaphore, nullptr);
	}

	for (size_t i = 0; i < (size_t)vk::MAX_FRAMES_IN_FLIGHT; i++)
	{
		vkFreeCommandBuffers(context.device, m_commandPool[i], 1, &m_commandBuffers[i]);
	}

	for (size_t i = 0; i < (size_t)vk::MAX_FRAMES_IN_FLIGHT; i++)
	{
		vkDestroyCommandPool(context.device, m_commandPool[i], nullptr);
	}
}

void vk::Renderer::CreateResources()
{
	CreateFences();
	CreateSemaphores();
	CreateCommandPool();
	AllocateCommandBuffers();
}

void vk::Renderer::CreateFences()
{
	for (size_t i = 0; i < (size_t)vk::MAX_FRAMES_IN_FLIGHT; i++)
	{
		// Fence
		VkFenceCreateInfo fenceInfo{
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			.flags = VK_FENCE_CREATE_SIGNALED_BIT
		};

		VkFence fence = VK_NULL_HANDLE;
		VK_CHECK(vkCreateFence(context.device, &fenceInfo, nullptr, &fence), "Failedd to create Fence.");
		m_Fences.push_back(std::move(fence));
	}
}

void vk::Renderer::CreateSemaphores()
{
	// Image available semaphore
	for (size_t i = 0; i < (size_t)vk::MAX_FRAMES_IN_FLIGHT; i++) {
		VkSemaphoreCreateInfo semaphoreInfo = {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
		};

		VkSemaphore semaphore = VK_NULL_HANDLE;
		VK_CHECK(vkCreateSemaphore(context.device, &semaphoreInfo, nullptr, &semaphore), "Failed to create image available semaphore");
		m_imageAvailableSemaphores.push_back(std::move(semaphore));
	}

	// Render finished sempahore
	for (size_t i = 0; i < (size_t)vk::MAX_FRAMES_IN_FLIGHT; i++) {
		VkSemaphoreCreateInfo semaphoreInfo = {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
		};

		VkSemaphore semaphore = VK_NULL_HANDLE;
		VK_CHECK(vkCreateSemaphore(context.device, &semaphoreInfo, nullptr, &semaphore), "Failed to create render finished semaphore");
		m_renderFinishedSemaphores.push_back(std::move(semaphore));
	}
}

void vk::Renderer::CreateCommandPool()
{
	for (size_t i = 0; i < (size_t)vk::MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkCommandPoolCreateInfo cmdPool{};
		cmdPool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		cmdPool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		cmdPool.queueFamilyIndex = context.graphicsFamilyIndex;

		VkCommandPool commandPool = VK_NULL_HANDLE;
		VK_CHECK(vkCreateCommandPool(context.device, &cmdPool, nullptr, &commandPool), "Failed to create command pool");
		m_commandPool.push_back(std::move(commandPool));
	}
}

void vk::Renderer::AllocateCommandBuffers()
{
	for (size_t i = 0; i < (size_t)vk::MAX_FRAMES_IN_FLIGHT; i++)
	{
		// Allocate command buffers from command pool
		VkCommandBufferAllocateInfo cmdAlloc{};
		cmdAlloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		cmdAlloc.commandPool = m_commandPool[i];
		cmdAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		cmdAlloc.commandBufferCount = 1;

		VkCommandBuffer cmd = VK_NULL_HANDLE;
		VK_CHECK(vkAllocateCommandBuffers(context.device, &cmdAlloc, &cmd), "Failed to allocate command buffer");
		m_commandBuffers.push_back(cmd);
	}
}

void vk::Renderer::Render()
{
	vkWaitForFences(context.device, 1, &m_Fences[vk::currentFrame], VK_TRUE, UINT64_MAX);

	uint32_t index;
	VkResult getImageIndex = vkAcquireNextImageKHR(context.device, context.swapchain, UINT64_MAX, m_imageAvailableSemaphores[vk::currentFrame], VK_NULL_HANDLE, &index);

	if (getImageIndex == VK_ERROR_OUT_OF_DATE_KHR)
	{
		// Recreate swapchain
		context.RecreateSwapchain();
		m_DepthPrepass->Resize();
		m_ShadowMap->Resize();
		m_ForwardPass->Resize();
		m_CompositePass->Resize();
		m_PresentPass->Resize();
	}
	else if (getImageIndex != VK_SUCCESS && getImageIndex != VK_SUBOPTIMAL_KHR)
	{
		throw std::runtime_error("Failed to aquire swapchain image");
	}

	vkResetFences(context.device, 1, &m_Fences[vk::currentFrame]);
	vkResetCommandBuffer(m_commandBuffers[vk::currentFrame], 0);

	VkCommandBuffer& cmd = m_commandBuffers[vk::currentFrame];

	{
		VkCommandBufferBeginInfo beginInfo = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
		};

		VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo), "Failed to begin command buffer");

		m_RayPass->Execute(cmd);
		m_ShadowMap->Execute(cmd);
		m_DepthPrepass->Execute(cmd);
		m_ForwardPass->Execute(cmd);
		m_CompositePass->Execute(cmd);
		m_PresentPass->Execute(cmd, index);

		vkEndCommandBuffer(cmd);
	}

	Submit();
	Present(index);

	vk::currentFrame = (vk::currentFrame + 1) % vk::MAX_FRAMES_IN_FLIGHT;
}

void vk::Renderer::Submit()
{
	VkPipelineStageFlags waitStage = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

	VkSubmitInfo subtmitInfo = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = &m_imageAvailableSemaphores[vk::currentFrame],
		.pWaitDstStageMask = &waitStage,
		.commandBufferCount = 1,
		.pCommandBuffers = &m_commandBuffers[vk::currentFrame],
		.signalSemaphoreCount = 1,
		.pSignalSemaphores = &m_renderFinishedSemaphores[vk::currentFrame]
	};

	VkResult result = vkQueueSubmit(context.graphicsQueue, 1, &subtmitInfo, m_Fences[vk::currentFrame]);

	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to submit command buffers");
	}

}

void vk::Renderer::Present(uint32_t imageIndex)
{
	VkPresentInfoKHR presentInfo = {
		.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = &m_renderFinishedSemaphores[vk::currentFrame],
		.swapchainCount = 1,
		.pSwapchains = &context.swapchain,
		.pImageIndices = &imageIndex,
	};


	VkResult result = vkQueuePresentKHR(context.presentQueue, &presentInfo);

	if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
	{
		// Recreate the swapchain
		context.RecreateSwapchain();
		m_DepthPrepass->Resize();
		m_ShadowMap->Resize();
		m_ForwardPass->Resize();
		m_CompositePass->Resize();
		m_PresentPass->Resize();
	}
}

void vk::Renderer::Update(double deltaTime)
{
	m_camera->Update(context.window, context.extent.width, context.extent.height, deltaTime);
	m_scene->Update(context.window);

	// Update passes
	m_ShadowMap->Update();
	m_ForwardPass->Update();
	m_PresentPass->Update();
}

void vk::Renderer::glfwHandleKeyboard(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	auto camera = static_cast<Camera*>(glfwGetWindowUserPointer(window));
	assert(camera);

	const bool isReleased = (GLFW_RELEASE == action); // check if key is released -> if not, its being held down

	switch (key)
	{
		case GLFW_KEY_W:
			camera->inputMap[std::size_t(EInputState::FORWARD)] = !isReleased;
			break;

		case GLFW_KEY_S:
			camera->inputMap[std::size_t(EInputState::BACKWARD)] = !isReleased;
			break;

		case GLFW_KEY_A:
			camera->inputMap[std::size_t(EInputState::LEFT)] = !isReleased;
			break;
	
		case GLFW_KEY_D:
			camera->inputMap[std::size_t(EInputState::RIGHT)] = !isReleased;
			break;

		case GLFW_KEY_Q:
			camera->inputMap[std::size_t(EInputState::DOWN)] = !isReleased;
			break;
		case GLFW_KEY_E:
			camera->inputMap[std::size_t(EInputState::UP)] = !isReleased;
			break;

		case GLFW_KEY_LEFT_SHIFT:
			camera->inputMap[std::size_t(EInputState::FAST)] = !isReleased;
			break;
		case GLFW_KEY_LEFT_CONTROL:
			camera->inputMap[std::size_t(EInputState::SLOW)] = !isReleased;
			break;
		default:
			;
	}

	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
	{
		glfwSetWindowShouldClose(window, GLFW_TRUE);
	}

	if (glfwGetKey(window, GLFW_KEY_5) == GLFW_PRESS)
	{
		postProcessSettings.Enable = postProcessSettings.Enable == true ? false : true;
		const std::string result = postProcessSettings.Enable == true ? "Enabled" : "Disabled";
		std::cout << "Post process: " << result << std::endl;
	}
}

void vk::Renderer::glfwMouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
	auto camera = static_cast<Camera*>(glfwGetWindowUserPointer(window));
	assert(camera);

	if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
	{
		auto& flag = camera->inputMap[std::size_t(EInputState::MOUSING)];

		flag = !flag; // we're using mouse now 
		if (flag)
		{
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

		}
		else
		{
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}
	}
}

// Get the current mouse position 
void vk::Renderer::glfwCallbackMotion(GLFWwindow* window, double x, double y)
{
	auto camera = static_cast<Camera*>(glfwGetWindowUserPointer(window));
	assert(camera);

	camera->mouseX = float(x);
	camera->mouseY = float(y);
}
