$VertexShader
#version 460 core

layout (location = 0) in vec3 ObjectPositionVS;

layout(std140, binding = 1) uniform PerFrameBuffer
{
	mat4 VP;
};

layout (std140, binding = 2) uniform PerModelBuffer
{
	mat4 MVP;
};

void main()
{
	gl_Position = VP * vec4(ObjectPositionVS, 1.0);
}

$PixelShader
#version 460 core

out vec4 PixelColor;

void main()
{
	PixelColor = vec4(0.6, 0.3, 0.2, 1.0);
}