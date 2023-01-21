#include "PCH.h"
#include "OpenGLRendererAPI.h"

#include "Tigraf/Renderer/Buffers/OpenGLBuffer.h"
#include "Tigraf/Renderer/Mesh/MeshPrimitives.h"
#include "Tigraf/Renderer/Shaders/Shader.h"

namespace Tigraf
{
	void OpenGLRendererAPI::init()
	{
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		glEnable(GL_DEPTH_TEST);
		glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

		glProvokingVertex(GL_LAST_VERTEX_CONVENTION);

		//glEnable(GL_CULL_FACE);
		//glCullFace(GL_BACK);

		glClearColor(0.2f, 0.6f, 0.8f, 1.0f);

		initGlobalUniformBuffers();

		MeshPrimitives::s_PrimitiveShader = Shader::create("resources\\shaders\\PrimitiveShader.glsl");
	}

	void OpenGLRendererAPI::clear()
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
	}

	void OpenGLRendererAPI::setClearColor(const glm::vec4& color)
	{
		glClearColor(color.r, color.g, color.b, color.a);
	}

	void OpenGLRendererAPI::setViewport(uint32_t x, uint32_t y, uint32_t width, uint32_t height)
	{
		glViewport(x, y, width, height);
	}

	void OpenGLRendererAPI::drawTriangles(const Ref<VertexBuffer>& vertexBuffer)
	{
		glBindVertexArray(reinterpret_cast<OpenGLVertexBuffer*>(vertexBuffer.get())->getVertexArrayID());
		glDrawArrays(GL_TRIANGLES, 0, vertexBuffer->getVertexCount());
	}

	void OpenGLRendererAPI::drawTrianglesIndexed(const Ref<VertexBuffer>& vertexBuffer)
	{
		glBindVertexArray(reinterpret_cast<OpenGLVertexBuffer*>(vertexBuffer.get())->getVertexArrayID());

		TIGRAF_ASSERT(vertexBuffer->getIndexBuffer() != nullptr, "Index Buffer must be set before drawing indexed!");

		glDrawElements(GL_TRIANGLES, vertexBuffer->getIndexBuffer()->getIndicesCount(), GL_UNSIGNED_INT, nullptr);
	}

	void OpenGLRendererAPI::drawPoints(const Ref<VertexBuffer>& vertexBuffer)
	{
		glBindVertexArray(reinterpret_cast<OpenGLVertexBuffer*>(vertexBuffer.get())->getVertexArrayID());
		glDrawArrays(GL_POINTS, 0, vertexBuffer->getVertexCount());
	}

	void OpenGLRendererAPI::drawPointsIndexed(const Ref<VertexBuffer>& vertexBuffer)
	{
		glBindVertexArray(reinterpret_cast<OpenGLVertexBuffer*>(vertexBuffer.get())->getVertexArrayID());

		TIGRAF_ASSERT(vertexBuffer->getIndexBuffer() != nullptr, "Index Buffer must be set before drawing indexed!");

		glDrawElements(GL_POINTS, vertexBuffer->getIndexBuffer()->getIndicesCount(), GL_UNSIGNED_INT, nullptr);
	}

	void OpenGLRendererAPI::drawPointsInstanced(const Ref<VertexBuffer>& vertexBuffer, uint32_t numInstances)
	{
		glBindVertexArray(reinterpret_cast<OpenGLVertexBuffer*>(vertexBuffer.get())->getVertexArrayID());
		glDrawArraysInstanced(GL_POINTS, 0, vertexBuffer->getVertexCount(), numInstances);
	}

	void OpenGLRendererAPI::initGlobalUniformBuffers()
	{
		UniformBuffer::s_TextureBuffer = UniformBuffer::create
		(
			NULL, 
			TEXTURE_HANDLE_OFFSET * (TEXTURE_2D_COUNT + TEXTURE_3D_COUNT + TEXTURE_CUBE_COUNT),
			//TODO(P3RK4N): Generalize bit flags
			GL_DYNAMIC_STORAGE_BIT
		);

		UniformBuffer::s_PerFrameBuffer = UniformBuffer::create
		(
			NULL,
			PER_FRAME_UNIFORM_BUFFER_SIZE,
			GL_DYNAMIC_STORAGE_BIT
		);

		UniformBuffer::s_PerModelBuffer = UniformBuffer::create
		(
			NULL,
			PER_MODEL_UNIFORM_BUFFER_SIZE,
			GL_DYNAMIC_STORAGE_BIT
		);

		
		GLuint textureBufferID = reinterpret_cast<OpenGLUniformBuffer*>(UniformBuffer::s_TextureBuffer.get())->getUniformBufferID();
		glBindBufferBase(GL_UNIFORM_BUFFER, TEXTURE_UNIFORM_BUFFER, textureBufferID);

		textureBufferID = reinterpret_cast<OpenGLUniformBuffer*>(UniformBuffer::s_PerFrameBuffer.get())->getUniformBufferID();
		glBindBufferBase(GL_UNIFORM_BUFFER, PER_FRAME_UNIFORM_BUFFER, textureBufferID);

		textureBufferID = reinterpret_cast<OpenGLUniformBuffer*>(UniformBuffer::s_PerModelBuffer.get())->getUniformBufferID();
		glBindBufferBase(GL_UNIFORM_BUFFER, PER_MODEL_UNIFORM_BUFFER, textureBufferID);

	}
}