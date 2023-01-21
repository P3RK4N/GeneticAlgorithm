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

#define MAX_POINTS 8
#define MAX_BONES 32
#define MAX_MUSCLES 32

layout(std430, binding = 0) buffer PointBuffer0
{
	Point[] Points0;
};

layout(std430, binding = 2) buffer LengthBuffer0
{
	float[] MuscleLengths0;
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
	float[] Distances;
};

//From 0 is good ones, from InstancesCount/2 is bad ones;
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

layout(binding = 0, offset = 0) uniform atomic_uint GoodInstances;
layout(binding = 0, offset = 4) uniform atomic_uint BadInstances;

void main()
{
	uint instanceOffset = uint(gl_GlobalInvocationID.x);
	float instanceLength = Distances[2 * instanceOffset];

	if(instanceLength >= HighLimit) 
	{
		FitnessIndices[atomicCounterIncrement(GoodInstances)] = instanceOffset;
	}
	else if(instanceLength <= LowLimit)
	{
		atomicCounterIncrement(BadInstances);
	}
};