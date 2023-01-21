#include "PCH.h"
#include "glslComputeShader.h"

#include "Tigraf/Renderer/Buffers/OpenGLBuffer.h"

#include <glm/gtc/type_ptr.hpp>

namespace Tigraf
{
	const static std::unordered_map<std::string, GLuint> NameToShaderType
	{
		{"ComputeShader", GL_COMPUTE_SHADER}
	};

#define GET_SHADER_TYPE(ShaderText) NameToShaderType[ShaderText.substr(0, ShaderText.size()-1)]

	glslComputeShader::glslComputeShader(const std::filesystem::path& path)
	{
		std::ifstream shaderFile(path);
		TIGRAF_ASSERT(shaderFile.is_open(), "Shader file could not be opened!");

		m_ShaderID = glCreateProgram();

		auto attachShader = [&](const std::string& shaderName, const GLchar* shaderSrc)
		{
			GLuint shader = glCreateShader(NameToShaderType.at(shaderName));

			glShaderSource(shader, 1, &shaderSrc, NULL);
			glCompileShader(shader);

			int  success;
			char infoLog[512];
			glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
			if(!success)
			{
				glGetShaderInfoLog(shader, 512, NULL, infoLog);
				CORE_ERROR(std::format("{} - {}", shader, infoLog));
			}
			glAttachShader(m_ShaderID, shader);
		};

		std::string currentShaderName = "";
		std::stringstream shaderSS;

		std::string line;
		while(std::getline(shaderFile, line))
		{
			if(line[0] == '$')
			{
				if(!currentShaderName.empty()) attachShader(currentShaderName, shaderSS.str().c_str());

				shaderSS = std::stringstream{};
				currentShaderName = line.substr(1, line.length() - 1);

				TIGRAF_ASSERT(NameToShaderType.find(currentShaderName) != NameToShaderType.end(), std::format("Unsupported Compute Shader: {0}", currentShaderName));
			}
			else shaderSS << line << '\n';
		}
		
		if(!currentShaderName.empty()) attachShader(currentShaderName, shaderSS.str().c_str());

		glLinkProgram(m_ShaderID);
	}

	glslComputeShader::~glslComputeShader()
	{
		glDeleteProgram(m_ShaderID);
	}

	void glslComputeShader::dispatch(int x, int y, int z)
	{
		//TODO: Remove all compute shaders?
	}

	void glslComputeShader::setFloat(float value, const char* name)
	{
		GLuint varID = glGetUniformLocation(m_ShaderID, name);
		glProgramUniform1f(m_ShaderID, varID, value);
	}

	void glslComputeShader::setFloat2(const glm::vec2& vector, const char* name)
	{
		GLuint varID = glGetUniformLocation(m_ShaderID, name);
		glProgramUniform2fv(m_ShaderID, varID, 1, glm::value_ptr(vector));
	}

	void glslComputeShader::setFloat3(const glm::vec3& vector, const char* name)
	{
		GLuint varID = glGetUniformLocation(m_ShaderID, name);
		glProgramUniform3fv(m_ShaderID, varID, 1, glm::value_ptr(vector));
	}

	void glslComputeShader::setFloat4(const glm::vec4& vector, const char* name)
	{
		GLuint varID = glGetUniformLocation(m_ShaderID, name);
		glProgramUniform4fv(m_ShaderID, varID, 1, glm::value_ptr(vector));
	}

	void glslComputeShader::setInt(int value, const char* name)
	{
		GLuint varID = glGetUniformLocation(m_ShaderID, name);
		glProgramUniform1i(m_ShaderID, varID, value);
	}

	void glslComputeShader::setInt2(const glm::i32vec2& vector, const char* name)
	{
		GLuint varID = glGetUniformLocation(m_ShaderID, name);
		glProgramUniform2iv(m_ShaderID, varID, 1, glm::value_ptr(vector));
	}

	void glslComputeShader::setInt3(const glm::i32vec3& vector, const char* name)
	{
		GLuint varID = glGetUniformLocation(m_ShaderID, name);
		glProgramUniform3iv(m_ShaderID, varID, 1, glm::value_ptr(vector));
	}

	void glslComputeShader::setInt4(const glm::i32vec4& vector, const char* name)
	{
		GLuint varID = glGetUniformLocation(m_ShaderID, name);
		glProgramUniform4iv(m_ShaderID, varID, 1, glm::value_ptr(vector));
	}

	void glslComputeShader::setMat3(const glm::mat3& mat, const char* name)
	{
		GLuint varID = glGetUniformLocation(m_ShaderID, name);
		glProgramUniformMatrix3fv(m_ShaderID, varID, 1, false, glm::value_ptr(mat));
	}

	void glslComputeShader::setMat4(const glm::mat4& mat, const char* name)
	{
		GLuint varID = glGetUniformLocation(m_ShaderID, name);
		glProgramUniformMatrix4fv(m_ShaderID, varID, 1, false, glm::value_ptr(mat));
	}
}