#include "Context.hpp"
#include "Buffer.hpp"
#include "Camera.hpp"
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>

#include "Utils.hpp"
#include <algorithm>

vk::Camera::Camera(Context& context, const glm::vec3 position, glm::vec3 direction, glm::vec3 up, float aspect) : context{ context }, m_position{ position }, m_direction{ direction }, m_up{ up }
{
	m_mouseSensitivity = 0.01f;
	m_increaseSpeed = 0.0f;
	m_transform.viewportSize = glm::vec2(context.extent.width, context.extent.height);
	m_transform.nearPlane = 0.1f;
	m_transform.farPlane = 100.0f;
	m_transform.fov = 45.0f;
	m_cameraSpeed = defaultSpeed;

	m_cameraUBO.resize(MAX_FRAMES_IN_FLIGHT);
	for (auto& buffer : m_cameraUBO)
	{
		buffer = CreateBuffer("cameraUBO", context, sizeof(CameraTransform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
	}
}

vk::Camera::~Camera()
{
	for (auto& buffer : m_cameraUBO)
	{
		buffer.Destroy(context.device);
	}
}

void vk::Camera::Update(GLFWwindow* window, uint32_t width, uint32_t height, double deltaTime)
{
	UpdateTransforms(width, height);

	// Write new data to the buffer to update uniform 
	VkDeviceSize size = sizeof(CameraTransform);
	m_cameraUBO[currentFrame].WriteToBuffer(m_transform, size);

	UpdateCameraRotation();
	UpdateCameraMovement();
}

void vk::Camera::UpdateTransforms(uint32_t width, uint32_t height)
{
	m_transform.model = glm::mat4(1.0f);
	m_transform.view = glm::lookAt(m_position, m_position + m_direction, m_up);
	m_transform.projection = glm::perspective(m_transform.fov, width / (float)height, m_transform.nearPlane, m_transform.farPlane);
	m_transform.projection[1][1] *= -1;
	m_transform.cameraPosition = glm::vec4(m_position.x, m_position.y, m_position.z, 1.0);
	m_transform.viewportSize = glm::vec2(width, height);
	m_transform.nearPlane = m_transform.nearPlane;
	m_transform.farPlane = m_transform.farPlane;
	m_transform.fov = m_transform.fov;
}

void vk::Camera::UpdateCameraMovement()
{
	if (inputMap[std::size_t(EInputState::FAST)])
	{
		m_cameraSpeed = std::min(m_cameraSpeed + m_speedIncreaseAmount * (float)deltaTime, maxSpeed);
		SetSpeed(m_cameraSpeed);
	}

	else if (inputMap[std::size_t(EInputState::SLOW)])
	{
		m_cameraSpeed = std::max(m_cameraSpeed - m_speedDecreaseAmount * (float)deltaTime, minSpeed);
		SetSpeed(m_cameraSpeed);
	}
	else {
		SetSpeed(defaultSpeed);
	}

	glm::vec3 movementAmount = glm::vec3(0.0f, 0.0f, 0.0f);

	if (inputMap[std::size_t(EInputState::FORWARD)])
	{
		glm::vec3 forward = glm::normalize(m_direction);
		movementAmount += forward;
	}

	if (inputMap[std::size_t(EInputState::BACKWARD)])
	{
		glm::vec3 forward = glm::normalize(m_direction);
		movementAmount -= forward;
	}

	if (inputMap[std::size_t(EInputState::LEFT)])
	{
		glm::vec3 rightVector = glm::normalize(glm::cross(m_direction, m_up));
		movementAmount -= rightVector;
	}

	if (inputMap[std::size_t(EInputState::RIGHT)])
	{
		glm::vec3 rightVector = glm::normalize(glm::cross(m_direction, m_up));
		movementAmount += rightVector;
	}

	if (inputMap[std::size_t(EInputState::UP)])
	{
		movementAmount += m_up;
	}

	if (inputMap[std::size_t(EInputState::DOWN)])
	{
		movementAmount -= m_up;
	}

	// Calculate the movement amount scaled by speed and delta time
	float speed = static_cast<float>(m_cameraSpeed * deltaTime);
	glm::vec3 targetPosition = m_position + (movementAmount * speed);

	m_position = targetPosition;

	//std::cout << "CamPos: " << m_position.x << ", " << m_position.y << ", " << m_position.z << std::endl;
}

void vk::Camera::UpdateCameraRotation()
{
	// If we're using the mouse
	if (inputMap[std::size_t(EInputState::MOUSING)])
	{
		// check if this is the first time mouse is being used, if so skip updating 
		// skip next frame so we have the correct lastx and lasty position for cursor
		if (wasMousing)
		{
			const auto sens = m_mouseSensitivity;
			const auto dx = sens * (mouseX - lastMouseX);
			const auto dy = sens * (lastMouseY - mouseY); // prevent inverted y

			UpdateCameraAngles(glm::vec2(dx, dy));
			UpdateCameraDirection();
		}

		lastMouseX = mouseX;
		lastMouseY = mouseY;
		wasMousing = true;
	}
	else
	{
		wasMousing = false;
	}
}

// Handling mouse movement based on learnings from 
/* Joey De Vries (2020). Learn OpenGL: Learn modern OpenGL graphics programming in a step-by-step fashion. Kendall & Welling. */
void vk::Camera::UpdateCameraAngles(const glm::vec2& offset)
{
	yaw += offset.x;
	pitch += offset.y;
	pitch = std::clamp(pitch, -89.0, 89.0);
}

// Compute the camera direction based on the cameras updated rotation 
void vk::Camera::UpdateCameraDirection()
{
	glm::vec3 direction = {
		static_cast<float>(cos(glm::radians(yaw)) * cos(glm::radians(pitch))),
		static_cast<float>(sin(glm::radians(pitch))),
		static_cast<float>(sin(glm::radians(yaw)) * cos(glm::radians(pitch)))
	};

	m_direction = glm::normalize(direction);
}

void vk::Camera::LogPosition() const
{
	std::cout << "Position: " << m_position.x << ", " << m_position.y << ", " << m_position.z << " Speed: " << m_cameraSpeed << std::endl;
}


