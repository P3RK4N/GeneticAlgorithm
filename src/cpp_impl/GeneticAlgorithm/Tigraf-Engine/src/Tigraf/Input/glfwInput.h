#pragma once
#include "Tigraf/Core/Application.h"
#include "Keycodes.h"

#include <GLFW/glfw3.h>

namespace Tigraf
{
	class glfwInput
	{
	public:
		static const bool isKeyDown(int keycode) 
		{	
			return glfwGetKey((GLFWwindow*)Application::s_Instance->getWindow()->getWindowHandle(), keycode) == GLFW_PRESS;		
		}

		static const bool isKeyUp(int keycode)
		{	
			return glfwGetKey((GLFWwindow*)Application::s_Instance->getWindow()->getWindowHandle(), keycode) == GLFW_RELEASE;	
		}

		static const bool isButtonDown(int keycode)
		{
			return glfwGetMouseButton((GLFWwindow*)Application::s_Instance->getWindow()->getWindowHandle(), keycode) == GLFW_PRESS;	
		}

		static const std::pair<int, int> getCursorPos()
		{
			double x, y;
			glfwGetCursorPos((GLFWwindow*)Application::s_Instance->getWindow()->getWindowHandle(), &x, &y);

			return { x, y };
		}

	};
}