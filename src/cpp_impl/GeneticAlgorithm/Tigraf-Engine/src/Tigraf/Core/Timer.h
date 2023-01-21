#pragma once
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace Tigraf
{
	class Application;

	struct TimeStep
	{
		double m_TotalTime = 0.0;
		double m_FrameTime = 0.0f;
	
		operator float() const { return m_FrameTime; }
	};

	class Timer
	{
	protected:
		TimeStep m_TimeStep;

		const TimeStep& getTimeStep() const { return m_TimeStep; }
		virtual const TimeStep& advance() = 0;
		friend class Application;
	};

	class glfwTimer : public Timer
	{
	public:
		virtual const TimeStep& advance() override
		{
			double totalTime = glfwGetTime();
			m_TimeStep.m_FrameTime = totalTime - m_TimeStep.m_TotalTime;
			m_TimeStep.m_TotalTime = totalTime;
			return m_TimeStep;
		}
	};
}