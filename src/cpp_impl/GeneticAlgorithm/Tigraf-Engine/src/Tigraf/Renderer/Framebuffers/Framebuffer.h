#pragma once
#include "Tigraf/Renderer/Textures/Texture.h"

#include <glm/glm.hpp>

namespace Tigraf
{
	class Framebuffer
	{
	public:
		Framebuffer(uint32_t width, uint32_t height)
			: m_Width(width), m_Height(height) {}

		virtual ~Framebuffer() {}

		virtual void bind() = 0;
		virtual void unbind() = 0;

		virtual void invalidate() = 0;
		virtual void resize(uint32_t width, uint32_t height) = 0;

		virtual void clearDepthStencil() = 0;
		virtual void clearColor(uint32_t index) = 0;
		virtual void clearAll() = 0;

		virtual void attachColorTexture(TextureFormat colorFormat) = 0;
		virtual void attachDepthStencilTexture(TextureFormat depthStencilFormat) = 0;

		const Ref<Texture2D>& getColorTexture(uint32_t textureIndex) { TIGRAF_ASSERT(textureIndex < m_ColorTextures.size(), "Invalid index!"); return m_ColorTextures[textureIndex]; }
		const Ref<Texture2D>& getDepthStencilTexture() { TIGRAF_ASSERT(m_DepthStencilTexture, "DepthStencil not attached!"); return m_DepthStencilTexture; }

		const float getWidth() { return m_Width; }
		const float getHeight() { return m_Height; }

	public:
		static Ref<Framebuffer> create(uint32_t width, uint32_t height);

		static float		s_ClearColor1f;
		static glm::vec2	s_ClearColor2f;
		static glm::vec3	s_ClearColor3f;
		static glm::vec4	s_ClearColor4f;

		static int			s_ClearColor1i;
		static glm::ivec2	s_ClearColor2i;
		static glm::ivec3	s_ClearColor3i;
		static glm::ivec4	s_ClearColor4i;

		static void setClearColor1f(float col)				{ s_ClearColor1f = col; }
		static void setClearColor2f(const glm::vec2& col)	{ s_ClearColor2f = col; }
		static void setClearColor3f(const glm::vec3& col)	{ s_ClearColor3f = col; }
		static void setClearColor4f(const glm::vec4& col)	{ s_ClearColor4f = col; }
		static void setClearColor1i(int col)				{ s_ClearColor1i = col; }
		static void setClearColor2i(const glm::ivec2& col)	{ s_ClearColor2i = col; }
		static void setClearColor3i(const glm::ivec3& col)	{ s_ClearColor3i = col; }
		static void setClearColor4i(const glm::ivec4& col)	{ s_ClearColor4i = col; }

	protected:
		std::vector<Ref<Texture2D>> m_ColorTextures{};
		Ref<Texture2D> m_DepthStencilTexture = nullptr;

		uint32_t m_Width = 0;
		uint32_t m_Height = 0;

	};
}