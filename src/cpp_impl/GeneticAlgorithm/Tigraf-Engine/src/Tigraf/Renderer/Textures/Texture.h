#pragma once
#include "Tigraf/Core/Core.h"

namespace Tigraf
{
	enum class TextureFormat : int
	{
		None,

		R8,				//Unsigned int 0-255	=> converted to 0.0f-1.0f in shader
		RG8,
		RGB8,
		RGBA8,

		R8I,			//Signed int -128 - 127 => stays same in shader 
		RG8I,
		RGB8I,
		RGBA8I,

		R32F,			//Float					=> stays same in shader
		RG32F,
		RGB32F,
		RGBA32F,

		R32I,			//Int					=> stays same in shader
		RG32I,
		RGB32I,
		RGBA32I,

		DEPTH24STENCIL8		//Depth and stencil texture => ????????
	};

	class Texture
	{
	public:
		virtual ~Texture() {}

		uint64_t getTextureHandle() { return m_TextureHandle; }
		TextureFormat getTextureFormat() { return m_TextureFormat; }

	protected:
		uint64_t m_TextureHandle = 0;
		TextureFormat m_TextureFormat = TextureFormat::None;
	};

	class Texture2D : public Texture
	{
	public:
		virtual ~Texture2D() {}

	public:
		static Ref<Texture2D> create(const char* filePath);
		static Ref<Texture2D> create(TextureFormat textureFormat, uint32_t width, uint32_t height, const void* textureData);

	protected:
		int m_Width = 0;
		int m_Height = 0;
	};

	class RWTexture2D : public Texture
	{
	public:
		virtual ~RWTexture2D() {}

	public:
		static Ref<RWTexture2D> create(TextureFormat textureFormat, uint32_t width, uint32_t height, const void* textureData);

	protected:
		int m_Width = 0;
		int m_Height = 0;
	};

	class TextureCube : public Texture
	{
	public:
		virtual ~TextureCube() {}

	public:
		/**
		* Creates Cubemap from 6 textures
		* 
		* @param baseFilePath contains directory path and file name (ex. folder1/workshop_front.jpg will be "folder1/workshop")
		* @param fileFormat is the format of picture (ex. workshop_front.jpg will be "jpg")
		* @return newly created cubemap texture pointer
		*/
		static Ref<TextureCube> create(const char* baseFilePath, const char* fileFormat);

		static const std::string s_CubeSides[6];

	protected:
		//Of each side
		int m_Width = 0;
		int m_Height = 0;
	};
}

