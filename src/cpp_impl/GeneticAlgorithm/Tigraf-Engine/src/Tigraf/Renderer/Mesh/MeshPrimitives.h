#pragma once
#include "Tigraf/Core/Core.h"

#include <glm/glm.hpp>

namespace Tigraf
{
	class Mesh;
	class Shader;

	class MeshPrimitives
	{
	public:
		static Ref<Mesh> Plane();
		static Ref<Mesh> Plane(const glm::mat4& initialTransform);

		static Ref<Mesh> Plane2D();
		static Ref<Mesh> Plane2D(const glm::mat4& initialTransform);

		static Ref<Mesh> Cube();
		static Ref<Mesh> Cube(const glm::mat4& initialTransform);

	private:
		static Ref<Shader> s_PrimitiveShader;

		friend class RendererAPI;
		friend class OpenGLRendererAPI;
	};
}