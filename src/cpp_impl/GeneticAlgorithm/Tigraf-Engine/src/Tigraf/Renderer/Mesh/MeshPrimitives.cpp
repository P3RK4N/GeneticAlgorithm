#include "PCH.h"
#include "MeshPrimitives.h"

#include "Mesh.h"
#include "Tigraf/Renderer/Buffers/Buffer.h"
#include "Tigraf/Renderer/Shaders/Shader.h"

namespace Tigraf
{
	struct PrimitiveVertex
	{
		glm::vec3 Position;
	};

	struct PrimitiveVertex2D
	{
		glm::vec2 Position;
	};

	struct TextureVertex2D
	{
		glm::vec2 Position;
		glm::vec2 TexCoord;
	};

	Ref<Shader> MeshPrimitives::s_PrimitiveShader = nullptr;

	Ref<Mesh> MeshPrimitives::Cube()
	{
		glm::mat4 idn = glm::mat4(1.0f);
		return MeshPrimitives::Cube(idn);
	}

	Ref<Mesh> MeshPrimitives::Plane()
	{
		glm::mat4 idn = glm::mat4(1.0f);
		return MeshPrimitives::Plane(idn);
	}

	Ref<Mesh> MeshPrimitives::Plane2D()
	{
		glm::mat4 idn = glm::mat4(1.0f);
		return MeshPrimitives::Plane2D(idn);
	}

	Ref<Mesh> MeshPrimitives::Cube(const glm::mat4& initialTransform)
	{
		std::vector<PrimitiveVertex> vertices =
		{
			{{ -0.5f, -0.5f, -0.5f }},		//0		
			{{ -0.5f, -0.5f,  0.5f }},		//1		
			{{ -0.5f,  0.5f, -0.5f }},		//2		
			{{ -0.5f,  0.5f,  0.5f }},		//3		
			{{  0.5f, -0.5f, -0.5f }},		//4		
			{{  0.5f, -0.5f,  0.5f }},		//5		
			{{  0.5f,  0.5f, -0.5f }},		//6
			{{  0.5f,  0.5f,  0.5f }}		//7
		};

		for(int index = 0; index < vertices.size(); index++)
		{
			vertices[index].Position = glm::vec3(initialTransform * glm::vec4(vertices[index].Position, 1.0f));
		};

		Ref<VertexBuffer> vertexBuffer = VertexBuffer::create(8, 12, vertices.data(), 0);
		vertexBuffer->pushVertexAttribute(VertexAttributeType::FLOAT3);

		std::vector<uint32_t> indices =
		{
			0,2,1,
			2,3,1,

			1,3,5,
			3,7,5,

			0,1,4,
			1,5,4,

			3,2,7,
			2,6,7,

			5,7,4,
			7,6,4,

			4,6,0,
			6,2,0
		};

		Ref<IndexBuffer> indexBuffer = IndexBuffer::create(indices, 0);

		return createRef<Mesh>(vertexBuffer, indexBuffer, s_PrimitiveShader);
	}

	Ref<Mesh> MeshPrimitives::Plane(const glm::mat4& initialTransform)
	{
		std::vector<PrimitiveVertex> vertices =
		{
			{{ -0.5f, 0.0f, -0.5f }},
			{{ -0.5f, 0.0f,  0.5f }},
			{{  0.5f, 0.0f, -0.5f }},
			{{  0.5f, 0.0f,  0.5f }}
		};

		for(int index = 0; index < vertices.size(); index++)
		{
			vertices[index].Position = glm::vec3(initialTransform * glm::vec4(vertices[index].Position, 1.0f));
		};

		Ref<VertexBuffer> vertexBuffer = VertexBuffer::create(4, 12, vertices.data(), 0);
		vertexBuffer->pushVertexAttribute(VertexAttributeType::FLOAT3);

		std::vector<uint32_t> indices =
		{
			0,2,1,
			2,3,1
		};

		Ref<IndexBuffer> indexBuffer = IndexBuffer::create(indices, 0);

		return createRef<Mesh>(vertexBuffer, indexBuffer, s_PrimitiveShader);
	}



	Ref<Mesh> MeshPrimitives::Plane2D(const glm::mat4& initialTransform)
	{

		std::vector<TextureVertex2D> vertices =
		{
			{{ -0.5f, -0.5f	}, {0.0f, 0.0f} },
			{{ -0.5f, 0.5f } , {0.0f, 1.0f} },
			{{  0.5f, -0.5f} , {1.0f, 0.0f} },
			{{  0.5f, 0.5f } , {1.0f, 1.0f} }
		};

		for(int index = 0; index < vertices.size(); index++)
		{
			vertices[index].Position = glm::vec2(initialTransform * glm::vec4(vertices[index].Position, 1.0f, 1.0f));
		};

		Ref<VertexBuffer> vertexBuffer = VertexBuffer::create(4, 16, vertices.data(), 0);
		vertexBuffer->pushVertexAttribute(VertexAttributeType::FLOAT2);
		vertexBuffer->pushVertexAttribute(VertexAttributeType::FLOAT2);

		std::vector<uint32_t> indices =
		{
			0,2,1,
			2,3,1
		};

		Ref<IndexBuffer> indexBuffer = IndexBuffer::create(indices, 0);

		return createRef<Mesh>(vertexBuffer, indexBuffer, s_PrimitiveShader);
	}
}