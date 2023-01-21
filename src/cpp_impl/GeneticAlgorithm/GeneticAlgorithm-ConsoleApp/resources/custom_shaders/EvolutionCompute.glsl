$ComputeShader
#version 460 core
#extension GL_ARB_shader_storage_buffer_object : require
#extension NV_shader_atomic_float : require

#define COMPUTE_X 1024
#define COMPUTE_Y 1

layout(local_size_x = COMPUTE_X, local_size_y = COMPUTE_Y) in;

struct Point
{
	vec3 Position;
	vec3 Velocity;
};

#define SPIDER 0

#if SPIDER
	#define MAX_POINTS 9
#else
	#define MAX_POINTS 8
#endif

#define MAX_BONES 32
#define MAX_MUSCLES 32

#define LAYER_SIZE 32

#define GAUSS_STD 1.0

layout(std430, binding = 0) buffer PointBuffer0
{
	Point[] Points0;
};

layout(std430, binding = 2) buffer LengthBuffer0
{
	float[] MuscleLengths0;
};

layout(std430, binding = 4) buffer NeuralLayersBuffer0
{
	float[] LayerValues0;
};

layout(std430, binding = 7) buffer NeuralLayersBuffer1
{
	float[] LayerValues1;
};

layout(std430, binding = 5) buffer StatsBuffer0
{
	uint InstancesCount;
	float DistancesSum;
	float SquaredDistancesSum;
	float DistancesMean;
	float DistancesVariance;
	float LowLimit;
	float HighLimit;
	uint LayerReadBuffer;
	float[] Distances;
};

layout(std140, binding = 1) uniform PerFrameBuffer
{
	mat4 CameraViewProjection;
	vec3 CameraWorldPosition;
	float TotalTime;
	float FrameTime;
};

//Stores indices of good instances
layout(std430, binding = 6) buffer FitnessIndicesBuffer0
{
	uint[] FitnessIndices;
};

layout(std140, binding = 3) uniform ShapeDataBuffer
{
	uint CurrentFrame;
	uint PointsCount;
	uint MusclesCount;
	uint BonesCount;

	ivec4[(MAX_BONES + MAX_MUSCLES) / 2] Connections;
	vec4[MAX_BONES / 4] BoneLengths;
	vec4[MAX_MUSCLES / 4] DefaultMuscleLengths;
	vec4[MAX_POINTS] DefaultPointPositions;
};

float rand(vec2 n)
{
	return fract(sin(dot(n.xy, vec2(12.9898, 78.233)))* 43758.5453) * 2.0 - 1.0;
}

//For FitnessCPU version
#define TOP 0.05
#define BOTTOM_REPLACE 0.3
#define BOTTOM_NOISE 0.8

void main()
{
	uint instanceIndex = FitnessIndices[uint(gl_GlobalInvocationID.x)];
	float percentile = float(gl_GlobalInvocationID.x+1) / float(InstancesCount);
	uint replacerIndex = instanceIndex;

	if(percentile <= BOTTOM_REPLACE)
	{
		uint modulo = uint(TOP * float(InstancesCount));
		replacerIndex = FitnessIndices[InstancesCount - 1 - uint(gl_GlobalInvocationID.x) % modulo];
	}


	const uint N = 2 * 3 * PointsCount;
	
	uint layer1 = N * LAYER_SIZE;
	uint layer2 = LAYER_SIZE * LAYER_SIZE;
	uint layer3 = LAYER_SIZE * MusclesCount;
	uint bias1 = LAYER_SIZE;
	uint bias2 = LAYER_SIZE;
	uint bias3 = MusclesCount;

	uint genesSize = (layer1 + layer2 + layer3 + bias1 + bias2 + bias3);
	uint currentLayerOffset = instanceIndex * genesSize;
	uint replacerLayerOffset = replacerIndex * genesSize;

	if(percentile >= BOTTOM_NOISE)
	{
		if(LayerReadBuffer == 1)
			for(uint i = 0; i < genesSize; i++)
				LayerValues1[currentLayerOffset + i] = LayerValues0[replacerLayerOffset + i];
		else
			for(uint i = 0; i < genesSize; i++)
				LayerValues0[currentLayerOffset + i] = LayerValues1[replacerLayerOffset + i];
	}
	else
	{
		if(LayerReadBuffer == 1)
			for(uint i = 0; i < genesSize; i++)
				LayerValues1[currentLayerOffset + i] = LayerValues0[replacerLayerOffset + i] + GAUSS_STD * rand(vec2(fract(float(i + gl_GlobalInvocationID.x) * 3.14157832112943), fract(TotalTime)));
		else
			for(uint i = 0; i < genesSize; i++)
				LayerValues0[currentLayerOffset + i] = LayerValues1[replacerLayerOffset + i] + GAUSS_STD * rand(vec2(fract(float(i + gl_GlobalInvocationID.x) * 3.14157832112943), fract(TotalTime)));
	}
};