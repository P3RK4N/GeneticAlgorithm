#pragma once
#include "Shader.h"

#include <glad/glad.h>
#include <filesystem>

namespace Tigraf
{
	class glslShader : public Shader
	{
	public:
		glslShader(const std::filesystem::path& path);
		virtual ~glslShader();

		virtual void use() override;
		virtual void dispatch(int x, int y, int z) override;

	public:
		virtual void setFloat(float value, const char* name) override;				
		virtual void setFloat2(const glm::vec2& vector, const char* name) override;
		virtual void setFloat3(const glm::vec3& vector, const char* name) override;
		virtual void setFloat4(const glm::vec4& vector, const char* name) override;

		virtual void setInt(int value, const char* name) override;							
		virtual void setInt2(const glm::i32vec2& vector, const char* name) override;
		virtual void setInt3(const glm::i32vec3& vector, const char* name) override;
		virtual void setInt4(const glm::i32vec4& vector, const char* name) override;

		virtual void setMat3(const glm::mat3& mat, const char* name) override; 
		virtual void setMat4(const glm::mat4& mat, const char* name) override;

	protected:
		GLuint m_ShaderID = 0;

	};
}
