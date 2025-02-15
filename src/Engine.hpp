#pragma once

#include "Context.hpp"
#include "Renderer.hpp"
#include "Camera.hpp"
#include <memory>

namespace vk
{
	class Engine
	{
	public:
		Engine();
		bool Initialize();
		void Run();
		void Shutdown();

	private:

		Context m_context;
		bool m_isRunning;
		double m_lastFrameTime;

		void Update(double deltaTime);
		void Render();

		std::unique_ptr<Renderer> m_Renderer;
	};
}