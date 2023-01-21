#pragma once

namespace Tigraf
{
	class GraphicsContext
	{
	public:
		GraphicsContext(std::function<void()> init, std::function<void()> swapBuffers) 
			: init(init), swapBuffers(swapBuffers) {}
		GraphicsContext() = default;

		std::function<void()> init = nullptr;
		std::function<void()> swapBuffers = nullptr;
	};
}
