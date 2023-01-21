#include "PCH.h"
#include "OpenGLTexture.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace Tigraf
{
	GLuint textureFormatToOpenGLDataFormat(TextureFormat textureFormat)
	{
		switch(textureFormat)
		{
		case TextureFormat::R8:
			return GL_RED;
		case TextureFormat::RG8:
			return GL_RG;
		case TextureFormat::RGB8:
			return GL_RGB;
		case TextureFormat::RGBA8:
			return GL_RGBA;

		case TextureFormat::R8I:
			return GL_RED_INTEGER;
		case TextureFormat::RG8I:
			return GL_RG_INTEGER;
		case TextureFormat::RGB8I:
			return GL_RGB_INTEGER;
		case TextureFormat::RGBA8I:
			return GL_RGBA_INTEGER;

		case TextureFormat::R32F:
			return GL_RED;
		case TextureFormat::RG32F:
			return GL_RG;
		case TextureFormat::RGB32F:
			return GL_RGB;
		case TextureFormat::RGBA32F:
			return GL_RGBA;

		case TextureFormat::R32I:
			return GL_RED_INTEGER;
		case TextureFormat::RG32I:
			return GL_RG_INTEGER;
		case TextureFormat::RGB32I:
			return GL_RGB_INTEGER;
		case TextureFormat::RGBA32I:
			return GL_RGBA_INTEGER;

		case TextureFormat::DEPTH24STENCIL8:
			return GL_DEPTH_STENCIL;
		}

		TIGRAF_ASSERT(0, "Invalid Texture Format");
	}

	GLuint textureFormatToOpenGLInternalFormat(TextureFormat textureFormat)
	{
		switch(textureFormat)
		{
		case TextureFormat::R8:
			return GL_R8;
		case TextureFormat::RG8:
			return GL_RG8;
		case TextureFormat::RGB8:
			return GL_RGB8;
		case TextureFormat::RGBA8:
			return GL_RGBA8;

		case TextureFormat::R8I:
			return GL_R8I;
		case TextureFormat::RG8I:
			return GL_RG8I;
		case TextureFormat::RGB8I:
			return GL_RGB8I;
		case TextureFormat::RGBA8I:
			return GL_RGBA8I;

		case TextureFormat::R32F:
			return GL_R32F;
		case TextureFormat::RG32F:
			return GL_RG32F;
		case TextureFormat::RGB32F:
			return GL_RGB32F;
		case TextureFormat::RGBA32F:
			return GL_RGBA32F;

		case TextureFormat::R32I:
			return GL_R32I;
		case TextureFormat::RG32I:
			return GL_RG32I;
		case TextureFormat::RGB32I:
			return GL_RGB32I;
		case TextureFormat::RGBA32I:
			return GL_RGBA32I;

		case TextureFormat::DEPTH24STENCIL8:
			return GL_DEPTH24_STENCIL8;
		}

		TIGRAF_ASSERT(0, "Invalid Texture Format");
	}

	GLuint textureFormatToComponentType(TextureFormat textureFormat)
	{
		switch(textureFormat)
		{
		case TextureFormat::R8:
		case TextureFormat::RG8:
		case TextureFormat::RGB8:
		case TextureFormat::RGBA8:
			return GL_UNSIGNED_BYTE;

		case TextureFormat::R8I:
		case TextureFormat::RG8I:
		case TextureFormat::RGB8I:
		case TextureFormat::RGBA8I:
			return GL_BYTE;

		case TextureFormat::R32I:
		case TextureFormat::RG32I:
		case TextureFormat::RGB32I:
		case TextureFormat::RGBA32I:
			return GL_INT;

		case TextureFormat::R32F:
		case TextureFormat::RG32F:
		case TextureFormat::RGB32F:
		case TextureFormat::RGBA32F:
			return GL_FLOAT;

		case TextureFormat::DEPTH24STENCIL8:
			return GL_UNSIGNED_INT_24_8;
		}

		TIGRAF_ASSERT(0, "Invalid Texture Format");
	}

	OpenGLTexture2D::OpenGLTexture2D(const char* filePath)
	{
		glCreateTextures(GL_TEXTURE_2D, 1, &m_TextureID);

		int channels;

		stbi_set_flip_vertically_on_load(1);
		uint8_t* data = stbi_load(filePath, &m_Width, &m_Height, &channels, 0);

		GLenum internalFormat, dataFormat;

		if(channels == 3)
		{
			internalFormat = GL_RGB8;
			dataFormat = GL_RGB;
			m_TextureFormat = TextureFormat::RGB8;
		}
		else
		{
			internalFormat = GL_RGBA8;
			dataFormat = GL_RGBA;
			m_TextureFormat = TextureFormat::RGBA8;
		}

		glTextureStorage2D(m_TextureID, 1, internalFormat, m_Width, m_Height);

		glTextureParameteri(m_TextureID, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTextureParameteri(m_TextureID, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		
		glTextureParameteri(m_TextureID, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTextureParameteri(m_TextureID, GL_TEXTURE_WRAP_T, GL_REPEAT);
		
		glTextureSubImage2D(m_TextureID, 0, 0, 0, m_Width, m_Height, dataFormat, GL_UNSIGNED_BYTE, data);

		stbi_image_free(data);

		m_TextureHandle = glGetTextureHandleARB(m_TextureID);
		
		//TODO(P3RK4N): Remove
		glMakeTextureHandleResidentARB(m_TextureHandle);
	}

	OpenGLTexture2D::OpenGLTexture2D(TextureFormat textureFormat, uint32_t width, uint32_t height, const void* textureData)
	{
		glCreateTextures(GL_TEXTURE_2D, 1, &m_TextureID);

		GLenum internalFormat, dataFormat, componentType;
		dataFormat = textureFormatToOpenGLDataFormat(textureFormat);
		internalFormat = textureFormatToOpenGLInternalFormat(textureFormat);
		componentType = textureFormatToComponentType(textureFormat);

		m_Width = width;
		m_Height = height;
		m_TextureFormat = textureFormat;

		glTextureStorage2D(m_TextureID, 1, internalFormat, width, height);

		glTextureParameteri(m_TextureID, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTextureParameteri(m_TextureID, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		
		glTextureParameteri(m_TextureID, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTextureParameteri(m_TextureID, GL_TEXTURE_WRAP_T, GL_REPEAT);

		if(textureData) glTextureSubImage2D(m_TextureID, 0, 0, 0, width, height, dataFormat, componentType, textureData);

		m_TextureHandle = glGetTextureHandleARB(m_TextureID);

		//TODO(P3RK4N): Remove
		glMakeTextureHandleResidentARB(m_TextureHandle);
	}

	OpenGLTexture2D::~OpenGLTexture2D()
	{
		glDeleteTextures(1, &m_TextureID);
	}


	OpenGLTextureCube::OpenGLTextureCube(const char* baseFilePath, const char* fileFormat)
	{
		glCreateTextures(GL_TEXTURE_CUBE_MAP, 1, &m_TextureID);

		GLenum internalFormat = 0;
		GLenum dataFormat = 0;
		int channels;
		stbi_set_flip_vertically_on_load(0);
		for(int i = 0; i < 6; i++)
		{
			std::string facePath = baseFilePath + std::string("_") + TextureCube::s_CubeSides[i] + std::string(".") + fileFormat;
			uint8_t* data = stbi_load(facePath.c_str(), &m_Width, &m_Height, &channels, 0);

			if(!internalFormat)
			{
				if(channels == 3)
				{
					internalFormat = GL_RGB8;
					dataFormat = GL_RGB;
					m_TextureFormat = TextureFormat::RGB8;
				}
				else
				{
					internalFormat = GL_RGBA8;
					dataFormat = GL_RGBA;
					m_TextureFormat = TextureFormat::RGBA8;
				}
				glTextureStorage2D(m_TextureID, 1, internalFormat, m_Width, m_Height);
			}

			glTextureSubImage3D(m_TextureID, 0, 0, 0, i, m_Width, m_Height, 1, dataFormat, GL_UNSIGNED_BYTE, data);
			stbi_image_free(data);
		};

		glTextureParameteri(m_TextureID, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTextureParameteri(m_TextureID, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		
		glTextureParameteri(m_TextureID, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTextureParameteri(m_TextureID, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTextureParameteri(m_TextureID, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

		m_TextureHandle = glGetTextureHandleARB(m_TextureID);
	
		//TODO(P3RK4N): Remove
		glMakeTextureHandleResidentARB(m_TextureHandle);
	}

	OpenGLTextureCube::~OpenGLTextureCube()
	{
		glDeleteTextures(1, &m_TextureID);
	}

	OpenGLRWTexture2D::OpenGLRWTexture2D(TextureFormat textureFormat, uint32_t width, uint32_t height, const void* textureData)
	{
		glCreateTextures(GL_TEXTURE_2D, 1, &m_TextureID);

		GLenum internalFormat, dataFormat, componentType;

		dataFormat = textureFormatToOpenGLDataFormat(textureFormat);
		internalFormat = textureFormatToOpenGLInternalFormat(textureFormat);
		componentType = textureFormatToComponentType(textureFormat);

		m_Width = width;
		m_Height = height;
		m_TextureFormat = textureFormat;

		glTextureStorage2D(m_TextureID, 1, internalFormat, m_Width, m_Height);
		if(textureData) glTextureSubImage2D(m_TextureID, 0, 0, 0, width, height, dataFormat, componentType, textureData);

		m_TextureHandle = glGetImageHandleARB(m_TextureID, 0, false, 0, internalFormat);
		
		//TODO: Remove
		glMakeImageHandleResidentARB(m_TextureHandle, GL_READ_WRITE);
	}

	OpenGLRWTexture2D::~OpenGLRWTexture2D()
	{

	}
}
