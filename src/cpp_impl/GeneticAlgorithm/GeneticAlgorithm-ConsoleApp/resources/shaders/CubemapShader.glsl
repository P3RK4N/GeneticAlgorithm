$VertexShader
#version 460 core


layout (location = 0) in vec3 ObjectPositionVS;
layout (location = 1) in vec2 TextureCoordinateVS;

layout(std140, binding = 1) uniform PerFrameBuffer
{
	mat4 VP;
	vec3 CameraWorldPosition;
};

layout(std140, binding = 2) uniform PerModelBuffer
{
	mat4 M;
	mat4 MVP;
};

out vec3 ObjectPositionPS;

void main()
{
	ObjectPositionPS = ObjectPositionVS;
	gl_Position = VP * vec4(ObjectPositionVS + CameraWorldPosition, 1.0);
}



$PixelShader
#version 460 core
#extension GL_ARB_bindless_texture : require

layout(std140, binding = 0) uniform TextureBuffer
{
	sampler2D textures2D[100];
	sampler3D textures3D[20];
	samplerCube texturesCube[20];
};

in vec3 ObjectPositionPS;

out vec4 FragColor;

void main()
{
	FragColor = texture(texturesCube[0], ObjectPositionPS);
}