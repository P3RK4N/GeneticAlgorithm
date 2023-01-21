#pragma once
#include "Buffer.h"

#include <glad/glad.h>
namespace Tigraf
{
	static GLenum OpenGLVertexAttributeType(VertexAttributeType type)
	{
		switch(type)
		{
		case VertexAttributeType::INT:		return GL_INT;
		case VertexAttributeType::INT2:		return GL_INT;
		case VertexAttributeType::INT3:		return GL_INT;
		case VertexAttributeType::INT4:		return GL_INT;
		case VertexAttributeType::FLOAT:	return GL_FLOAT;
		case VertexAttributeType::FLOAT2:	return GL_FLOAT;
		case VertexAttributeType::FLOAT3:	return GL_FLOAT;
		case VertexAttributeType::FLOAT4:	return GL_FLOAT;
		case VertexAttributeType::FLOAT3X3:	return GL_FLOAT;
		case VertexAttributeType::FLOAT4X4:	return GL_FLOAT;
		}
	};

	static GLuint OpenGLVertexAttributeTypeSize(VertexAttributeType type)
	{
		switch(type)
		{
		case VertexAttributeType::INT:		return sizeof(int)   * 1;
		case VertexAttributeType::INT2:		return sizeof(int)   * 2;
		case VertexAttributeType::INT3:		return sizeof(int)   * 3;
		case VertexAttributeType::INT4:		return sizeof(int)   * 4;
		case VertexAttributeType::FLOAT:	return sizeof(float) * 1;
		case VertexAttributeType::FLOAT2:	return sizeof(float) * 2;
		case VertexAttributeType::FLOAT3:	return sizeof(float) * 3;
		case VertexAttributeType::FLOAT4:	return sizeof(float) * 4;
		case VertexAttributeType::FLOAT3X3:	return sizeof(float) * 9;
		case VertexAttributeType::FLOAT4X4:	return sizeof(float) * 16;
		}
	};

	static GLuint OpenGLVertexAttributeTypeCount(VertexAttributeType type)
	{
		switch(type)
		{
		case VertexAttributeType::INT:		return 1;
		case VertexAttributeType::INT2:		return 2;
		case VertexAttributeType::INT3:		return 3;
		case VertexAttributeType::INT4:		return 4;
		case VertexAttributeType::FLOAT:	return 1;
		case VertexAttributeType::FLOAT2:	return 2;
		case VertexAttributeType::FLOAT3:	return 3;
		case VertexAttributeType::FLOAT4:	return 4;
		case VertexAttributeType::FLOAT3X3:	return 9;
		case VertexAttributeType::FLOAT4X4:	return 16;
		}
	};

	class OpenGLVertexBuffer : public VertexBuffer
	{
	public:
		virtual ~OpenGLVertexBuffer();
		OpenGLVertexBuffer(GLuint vertexCount, GLuint vertexSize, void* data, GLuint flags = GL_DYNAMIC_STORAGE_BIT);

		virtual void pushVertexAttribute(VertexAttributeType type) override;
		virtual void setIndexBuffer(const Ref<IndexBuffer>& indexBuffer) override;

		const GLuint getVertexArrayID() const { return m_VertexArrayID; }
		const GLuint getVertexBufferID() const { return m_VertexBufferID; }

	protected:
		GLuint m_VertexBufferID = 0;
		GLuint m_VertexArrayID = 0;

		uint32_t m_AttributeOffset = 0;
		uint32_t m_AttributeCount = 0;
	};

	class OpenGLIndexBuffer : public IndexBuffer
	{
	public:
		virtual ~OpenGLIndexBuffer();
		OpenGLIndexBuffer(const std::vector<GLuint>& indices, GLuint flags);
		
		const GLuint getIndexBufferID() const { return m_IndexBufferID; }

	protected:
		GLuint m_IndexBufferID = 0;
	};

	class OpenGLUniformBuffer : public UniformBuffer
	{
	public:
		OpenGLUniformBuffer(void* data, uint32_t byteSize, GLuint storageFlags);
		virtual ~OpenGLUniformBuffer();

		virtual void updateBuffer(void* subData, uint32_t sizeInBytes, uint32_t byteOffset) override;
		virtual void bind(uint16_t bindIndex) override;
		virtual void unbind() override;

		uint32_t getUniformBufferID() { return m_UniformBufferID; }

	protected:
		GLuint m_UniformBufferID = 0;
	};

	class OpenGLRWBuffer : public RWBuffer
	{
	public:
		OpenGLRWBuffer(void* data, uint32_t sizeInBytes, GLuint storageFlags);
		virtual ~OpenGLRWBuffer();

		virtual void updateBuffer(void* subData, uint32_t sizeInBytes, uint32_t byteOffset) override;
		virtual void get(void* subData, uint32_t byteOffset, uint32_t sizeInBytes) override;
		virtual void bind(uint16_t bindIndex) override;
		virtual void unbind() override;

		uint32_t getRWBufferID() { return m_RWBufferID; }

	protected:
		GLuint m_RWBufferID = 0;
	};
}