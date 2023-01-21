#pragma once
//TODO(P3RK4N): Remove includes!
#include "RendererAPI.h"
#include "OpenGLRendererAPI.h"

namespace Tigraf
{
	enum class RendererAPIName : int
	{
		None,
		OpenGL,
		Vulkan,
		DirectX11,
		DirectX12
	};

	class Renderer
	{
	public:
		static RendererAPIName s_RendererAPIName;
		static RendererAPI* s_RendererAPI;

		static void init();
		static void shutdown();

	};
}