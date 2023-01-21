$VertexShader
#version 460 core

layout (location = 0) in vec2 ClipPositionVS;
layout (location = 1) in vec2 TextureCoordinateVS;

out vec2 TextureCoordinatePS;

void main()
{
	TextureCoordinatePS = TextureCoordinateVS;
	gl_Position = vec4(ClipPositionVS.xy, 0.0, 1.0);
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

layout(std140, binding = 3) uniform ImageBuffer
{
	layout(rgba8) image2D Result;
};

layout(std430, binding = 0) buffer RWBuffer
{
	vec4 color;
};

in vec2 TextureCoordinatePS;

layout (location = 0) out vec4 FragColor;

void main()
{
	vec2 size = imageSize(Result);

	vec2 coords = vec2(TextureCoordinatePS.x * size.x, TextureCoordinatePS.y * size.y);
	FragColor = imageLoad(Result, ivec2(coords));
}
