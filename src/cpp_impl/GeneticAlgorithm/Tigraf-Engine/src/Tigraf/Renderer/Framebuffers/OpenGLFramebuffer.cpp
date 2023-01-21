#include "PCH.h"
#include "OpenGLFramebuffer.h"

#include "Tigraf/Renderer/Textures/OpenGLTexture.h"
#include "Tigraf/Core/Application.h"

namespace Tigraf
{	
	OpenGLFramebuffer::OpenGLFramebuffer(uint32_t width, uint32_t height)
		: Framebuffer::Framebuffer(width, height)
	{
		glCreateFramebuffers(1, &m_FramebufferID);
	}

	OpenGLFramebuffer::~OpenGLFramebuffer()
	{
		glDeleteFramebuffers(1, &m_FramebufferID);
	}

	void OpenGLFramebuffer::attachColorTexture(TextureFormat colorFormat)
	{
		TIGRAF_ASSERT(colorFormat == TextureFormat::RGBA8, "Tigraf currently does not support this framebuffer color format!");

		Ref<Texture2D> colorTexture = Texture2D::create(colorFormat, m_Width, m_Height, nullptr);
		GLuint textureID = reinterpret_cast<OpenGLTexture2D*>(colorTexture.get())->getTextureID();

		glNamedFramebufferTexture(m_FramebufferID, GL_COLOR_ATTACHMENT0 + m_ColorTextures.size(), textureID, 0);

		m_ColorTextures.push_back(colorTexture);
	}

	void OpenGLFramebuffer::attachDepthStencilTexture(TextureFormat depthStencilFormat)
	{
		TIGRAF_ASSERT(depthStencilFormat == TextureFormat::DEPTH24STENCIL8, "Tigraf currently does not support this framebuffer depth stencil format!");
		TIGRAF_ASSERT(!m_DepthStencilTexture, "Depth stencil texture already attached!");

		Ref<Texture2D> depthStencilTexture = Texture2D::create(depthStencilFormat, m_Width, m_Height, nullptr);
		GLuint textureID = reinterpret_cast<OpenGLTexture2D*>(depthStencilTexture.get())->getTextureID();

		glNamedFramebufferTexture(m_FramebufferID, GL_DEPTH_STENCIL_ATTACHMENT, textureID, 0);

		m_DepthStencilTexture = depthStencilTexture;
	}

	void OpenGLFramebuffer::invalidate()
	{
		if(m_FramebufferID)
		{
			glDeleteFramebuffers(1, &m_FramebufferID);
			glCreateFramebuffers(1, &m_FramebufferID);
		}

		if(m_ColorTextures.size() > 0)
		{
			std::vector<Ref<Texture2D>> newColorTextures = m_ColorTextures;
			m_ColorTextures.clear();
			for(auto& colorTexture : newColorTextures)
			{
				attachColorTexture(colorTexture->getTextureFormat());
			}

			glNamedFramebufferDrawBuffers(m_FramebufferID, m_ColorTextures.size(), drawBuffers);
		}
		else glNamedFramebufferDrawBuffer(m_FramebufferID, GL_NONE);

		if(m_DepthStencilTexture)
		{
			TextureFormat depthStencilFormat = m_DepthStencilTexture->getTextureFormat();
			m_DepthStencilTexture = nullptr;
			attachDepthStencilTexture(depthStencilFormat);
		}

		CORE_TRACE(std::format("Framebuffer initialized: {0}", (int)(glCheckNamedFramebufferStatus(m_FramebufferID, GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE)));
	}

	void OpenGLFramebuffer::resize(uint32_t width, uint32_t height)
	{
		m_Width = width;
		m_Height = height;
		invalidate();
	}

	void OpenGLFramebuffer::clearDepthStencil()
	{
		if(m_DepthStencilTexture)
			glClearNamedFramebufferfi(m_FramebufferID, GL_DEPTH_STENCIL, 0, 1.0f, 0);
	}

	void OpenGLFramebuffer::clearColor(uint32_t index)
	{
		TIGRAF_ASSERT(m_ColorTextures.size() > index, "Invalid color texture index!");
		
		TextureFormat format = m_ColorTextures[index]->getTextureFormat();
		GLuint textureID = reinterpret_cast<OpenGLTexture2D*>(m_ColorTextures[index].get())->getTextureID();

		switch(format)
		{
		case TextureFormat::R8:
			glClearTexImage(textureID, 0, GL_RED, GL_FLOAT, &s_ClearColor1f);
			return;
		case TextureFormat::RG8:
			glClearTexImage(textureID, 0, GL_RG, GL_FLOAT, &s_ClearColor2f);
			return;
		case TextureFormat::RGB8:
			glClearTexImage(textureID, 0, GL_RGB, GL_FLOAT, &s_ClearColor3f);
			return;
		case TextureFormat::RGBA8:
			glClearTexImage(textureID, 0, GL_RGBA, GL_FLOAT, &s_ClearColor4f);
			return;

		case TextureFormat::R8I:
			glClearTexImage(textureID, 0, GL_RED_INTEGER, GL_INT, &s_ClearColor1i);
			return;
		case TextureFormat::RG8I:
			glClearTexImage(textureID, 0, GL_RED_INTEGER, GL_INT, &s_ClearColor2i);
			return;
		case TextureFormat::RGB8I:
			glClearTexImage(textureID, 0, GL_RED_INTEGER, GL_INT, &s_ClearColor3i);
			return;
		case TextureFormat::RGBA8I:
			glClearTexImage(textureID, 0, GL_RED_INTEGER, GL_INT, &s_ClearColor4i);
			return;

		}

		TIGRAF_ASSERT(0, "Invalid texture format!");
	}

	void OpenGLFramebuffer::clearAll()
	{
		clearDepthStencil();
		for(size_t index = 0; index < m_ColorTextures.size(); index++)
			clearColor(index);
	}

	void OpenGLFramebuffer::bind()
	{
		glBindFramebuffer(GL_FRAMEBUFFER, m_FramebufferID);
		glViewport(0, 0, m_Width, m_Height);
		clearAll();
	}

	void OpenGLFramebuffer::unbind()
	{
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		//TODO(P3RK4N): Change/Remove
		auto [width, height] = Application::s_Instance->getWindow()->getSize();
		glViewport(0, 0, width, height);
	}
}