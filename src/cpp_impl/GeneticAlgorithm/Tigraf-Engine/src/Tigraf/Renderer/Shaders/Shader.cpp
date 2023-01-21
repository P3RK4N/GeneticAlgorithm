#include "PCH.h"
#include "Shader.h"

#include "glslShader.h"

namespace Tigraf
{
	Ref<Shader> Shader::create(const std::filesystem::path& shaderPath)
	{
		return createRef<glslShader>(shaderPath);
	}

}