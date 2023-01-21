$VertexShader
#version 460 core

layout (location = 0) in vec3 ClipPositionVS;

out vec2 TextureCoordinatesPS;

void main()
{
	TextureCoordinatesPS = ClipPositionVS.xz / 2.0 + vec2(0.5,0.5);
	gl_Position = vec4(ClipPositionVS.xz, 0.0, 1.0);
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

in vec2 TextureCoordinatesPS;

layout (location = 0) out vec4 FragColor;

void main()
{
	vec3 color = texture(textures2D[0], TextureCoordinatesPS).rgb;
	FragColor = vec4(color,1.0);
}
