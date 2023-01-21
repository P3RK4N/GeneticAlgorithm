#include "PCH.h"
#include "ComputeShader.h"

#include "glslComputeShader.h"

namespace Tigraf
{
	Ref<ComputeShader> ComputeShader::create(const std::filesystem::path& shaderPath)
	{
		return createRef<glslComputeShader>(shaderPath);
	}

}