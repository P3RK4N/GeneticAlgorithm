#pragma once
#include "Texture.h"

#include <glad/glad.h>

namespace Tigraf
{
	class OpenGLTexture2D : public Texture2D
	{
	public:
		OpenGLTexture2D(const char* filePath);

		/**
		* NOTE: Each color component must have size of 1 byte
		*/
		OpenGLTexture2D(TextureFormat textureFormat, uint32_t width, uint32_t height, const void* textureData);

		virtual ~OpenGLTexture2D();

		const GLuint getTextureID() { return m_TextureID; }

	protected:
		GLuint m_TextureID = 0;
	};

	class OpenGLTextureCube : public TextureCube
	{
	public:
		OpenGLTextureCube(const char* baseFilePath, const char* fileFormat);
		virtual ~OpenGLTextureCube();

	protected:
		GLuint m_TextureID = 0;

	private:

	};

	class OpenGLRWTexture2D : public RWTexture2D
	{
	public:
		OpenGLRWTexture2D(TextureFormat textureFormat, uint32_t width, uint32_t height, const void* textureData);
		virtual ~OpenGLRWTexture2D();

		const GLuint getTextureID() { return m_TextureID; }

	protected:
		GLuint m_TextureID = 0;
	};
}
