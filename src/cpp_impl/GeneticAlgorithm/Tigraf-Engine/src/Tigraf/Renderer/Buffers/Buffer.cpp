#include "PCH.h"
#include "Buffer.h"

#include "OpenGLBuffer.h"

namespace Tigraf
{
	Ref<VertexBuffer> VertexBuffer::create(uint32_t vertexCount, uint32_t vertexSize, void* data, uint32_t storageFlags)
	{
		return createRef<OpenGLVertexBuffer>(vertexCount, vertexSize, data, storageFlags);
	}

	Ref<IndexBuffer> IndexBuffer::create(const std::vector<uint32_t>& indices, uint32_t storageFlags)
	{
		return createRef<OpenGLIndexBuffer>(indices, storageFlags);
	}

	Ref<UniformBuffer> UniformBuffer::create(void* data, uint32_t byteSize, uint32_t storageFlags)
	{
		return createRef<OpenGLUniformBuffer>(data, byteSize, storageFlags);
	}

	std::unordered_set<uint16_t> UniformBuffer::s_CurrentUniformBuffers = { 0, 1, 2 };

	Ref<UniformBuffer> UniformBuffer::s_TextureBuffer = nullptr;
	Ref<UniformBuffer> UniformBuffer::s_PerFrameBuffer = nullptr;
	Ref<UniformBuffer> UniformBuffer::s_PerModelBuffer = nullptr;

	std::unordered_set<uint16_t> RWBuffer::s_CurrentRWBuffers = {};

	Ref<RWBuffer> RWBuffer::create(void* data, uint32_t sizeInBytes, uint32_t storageFlags)
	{
		return createRef<OpenGLRWBuffer>(data, sizeInBytes, storageFlags);
	}
}