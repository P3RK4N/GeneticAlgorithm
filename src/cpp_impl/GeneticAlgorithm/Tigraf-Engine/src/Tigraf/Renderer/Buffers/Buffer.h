#pragma once
#include "Tigraf/Core/Core.h"
#include "UniformBufferDefines.h"
#include "RWBufferDefines.h"

#include <cstdint>
#include <unordered_set>

namespace Tigraf
{
#define isINT(VertexAttributeType)		(uint32_t)VertexAttributeType < 10U
#define isFLOAT(VertexAttributeType)	(uint32_t)VertexAttributeType < 20U && (uint32_t)VertexAttributeType >= 10U
#define isDOUBLE(VertexAttributeType)	(uint32_t)VertexAttributeType >= 20U

/**
* Sets texture to index location in UNIFORM_TEXTURE_BUFFER
*/
#define SET_TEXTURE_HANDLE(textureHandle, textureIndex)											\
	{																							\
		uint8_t textureHandleWrapper[16];														\
		uint64_t handle = textureHandle;														\
		memcpy(textureHandleWrapper, &handle, 8);												\
		UniformBuffer::s_TextureBuffer->updateBuffer(textureHandleWrapper, 16, textureIndex);	\
	}

/**
* Updates per frame buffer with given data,size and offset
*/
#define UPDATE_PER_FRAME_BUFFER(data, byteSize, byteOffset)	UniformBuffer::s_PerFrameBuffer->updateBuffer((void*)&(data), byteSize, byteOffset)

/**
* Updates per model buffer with given data,size and offset
*/
#define UPDATE_PER_MODEL_BUFFER(data, byteSize, byteOffset)	UniformBuffer::s_PerModelBuffer->updateBuffer((void*)&(data), byteSize, byteOffset)


	enum class VertexAttributeType : uint16_t
	{
		INT = 0,
		INT2 = 1,
		INT3 = 2,
		INT4 = 3,

		FLOAT = 10,
		FLOAT2 = 11,
		FLOAT3 = 12,
		FLOAT4 = 13,

		FLOAT3X3 = 14,
		FLOAT4X4 = 15,
	};

	class IndexBuffer;

	class VertexBuffer
	{
	public:
		virtual ~VertexBuffer() {}

		virtual void pushVertexAttribute(VertexAttributeType type) = 0;

		const Ref<IndexBuffer>& getIndexBuffer() { return m_IndexBuffer; }
		virtual void setIndexBuffer(const Ref<IndexBuffer>& indexBuffer) = 0;
		
		const uint32_t getVertexSize() const { return m_VertexSize; }
		const uint32_t getVertexCount() const { return m_VertexCount; }

	public:
		static Ref<VertexBuffer> create(uint32_t vertexCount, uint32_t vertexSize, void* data, uint32_t storageFlags);

	protected:
		uint32_t m_VertexSize = 0;
		uint32_t m_VertexCount = 0;

		Ref<IndexBuffer> m_IndexBuffer = nullptr;
	};

	class IndexBuffer
	{
	public:
		virtual ~IndexBuffer() {}

		const uint32_t getIndicesCount() const { return m_IndicesCount; }

	public:
		static Ref<IndexBuffer> create(const std::vector<uint32_t>& indices, uint32_t storageFlags);

	protected:
		uint32_t m_IndicesCount = 0;
	};

	class UniformBuffer
	{
	public:
		virtual ~UniformBuffer() {}

		virtual void updateBuffer(void* subData, uint32_t byteSize, uint32_t byteOffset) {}
		const uint16_t getBindIndex() { return m_BindIndex; }

		virtual void bind(uint16_t bindIndex) = 0;
		virtual void unbind() = 0;

	public:
		static Ref<UniformBuffer> create(void* data, uint32_t sizeInBytes, uint32_t storageFlags);

		static std::unordered_set<uint16_t> s_CurrentUniformBuffers;
		static Ref<UniformBuffer> s_TextureBuffer;
		static Ref<UniformBuffer> s_PerFrameBuffer;
		static Ref<UniformBuffer> s_PerModelBuffer;
		
	protected:
		uint32_t m_SizeInBytes = 0;
		uint16_t m_BindIndex = -1;
	};

	class RWBuffer
	{
	public:
		virtual ~RWBuffer() {}

		virtual void updateBuffer(void* subData, uint32_t byteSize, uint32_t byteOffset) = 0;
		virtual void get(void* subData, uint32_t byteOffset, uint32_t sizeInBytes) = 0;
		const uint16_t getBindIndex() { return m_BindIndex; }

		virtual void bind(uint16_t bindIndex) = 0;
		virtual void unbind() = 0;

	public:
		static Ref<RWBuffer> create(void* data, uint32_t sizeInBytes, uint32_t storageFlags);

		static std::unordered_set<uint16_t> s_CurrentRWBuffers;

	protected:
		uint32_t m_SizeInBytes = 0;
		uint16_t m_BindIndex = -1;
	};
}