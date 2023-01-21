$ComputeShader
#version 460 core
#extension GL_ARB_bindless_texture : require

layout(local_size_x = 16, local_size_y = 16) in;

layout(std140, binding = 3) uniform ImageBuffer
{
	layout(rgba8) image2D Result;
};

layout(std140, binding = 1) uniform PerFrameBuffer
{
	mat4 CameraViewProjection;
	vec3 CameraWorldPosition;
	float TotalTime;
};

void main()
{
	if(gl_GlobalInvocationID.x >= imageSize(Result).x || gl_GlobalInvocationID.y >= imageSize(Result).y) return;

	vec4 current = imageLoad(Result, ivec2(gl_GlobalInvocationID.xy));
	current.x = sin(TotalTime) / 2.0 + 0.5;
	imageStore(Result, ivec2(gl_GlobalInvocationID.xy), current);
};