$ComputeShader
#version 460 core
#extension GL_ARB_shader_storage_buffer_object : require

#define COMPUTE_X 1024
#define COMPUTE_Y 1

#define SPIDER 0

#if SPIDER
	#define MAX_POINTS 9
#else
	#define MAX_POINTS 8
#endif

#define MAX_BONES 32
#define MAX_MUSCLES 32

#define LAYER_SIZE 32
#define MAX_MUSCLE_OFFSET 0.2
#define MAX_MUSCLE_DIFF 0.15

layout(local_size_x = COMPUTE_X, local_size_y = COMPUTE_Y) in;

struct Point
{
	vec3 Position;
	vec3 Velocity;
};


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

struct Distance_Index
{
	float dist;
	uint ind;
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
	Distance_Index[] Distances;
};


void updateLengths0()
{
	//2 arrays of vec3 + 2 vec3
	const uint N = 2 * 3 * PointsCount;
	
	const uint layer1 = N * LAYER_SIZE;
	const uint layer2 = LAYER_SIZE * LAYER_SIZE;
	const uint layer3 = LAYER_SIZE * MusclesCount;
	const uint bias1 = LAYER_SIZE;
	const uint bias2 = LAYER_SIZE;
	const uint bias3 = MusclesCount;

	const uint instanceOffset = uint(gl_GlobalInvocationID.x) * (layer1+layer2+layer3+bias1+bias2+bias3);
	const uint pointsOffset = uint(gl_GlobalInvocationID.x) * PointsCount;
	const uint musclesOffset = uint(gl_GlobalInvocationID.x) * MusclesCount;

	const uint layer1offset = instanceOffset;
	const uint bias1offset = instanceOffset + layer1;

	const uint layer2offset = instanceOffset + layer1 + bias1;
	const uint bias2offset = instanceOffset + layer1 + bias1 + layer2;

	const uint layer3offset = instanceOffset + layer1 + bias1 + layer2 + bias2;
	const uint bias3offset = instanceOffset + layer1 + bias1 + layer2 + bias2 + layer3;

	//Layer 1
	float[LAYER_SIZE] layer1out;
	Point referencePoint = Points0[pointsOffset];
	//1xIN @ INx32 = 1x32;		ACTUALLY 1xIN @ 32xIN
	//output is array[32] where each array element is computed with 2 row vectors

	for(int outputIndex = 0; outputIndex < LAYER_SIZE; outputIndex++)
	{
		uint rowOffset = layer1offset + N * outputIndex;
		float val =
		LayerValues0[rowOffset + 0] * referencePoint.Position.x +
		LayerValues0[rowOffset + 1] * referencePoint.Position.y +
		LayerValues0[rowOffset + 2] * referencePoint.Position.z +
		LayerValues0[rowOffset + 3] * referencePoint.Velocity.x +
		LayerValues0[rowOffset + 4] * referencePoint.Velocity.y +
		LayerValues0[rowOffset + 5] * referencePoint.Velocity.z;

		//#pragma unroll
		for(uint i = 1; i < PointsCount; i ++)
		{
			vec3 posDiff = Points0[pointsOffset + i].Position - referencePoint.Position;
			vec3 velDiff = Points0[pointsOffset + i].Velocity - referencePoint.Velocity;

			uint rowIndex = rowOffset + 6 * i;
			val +=
				LayerValues0[rowIndex + 0] * posDiff.x +
				LayerValues0[rowIndex + 1] * posDiff.y +
				LayerValues0[rowIndex + 2] * posDiff.z +
				LayerValues0[rowIndex + 3] * velDiff.x +
				LayerValues0[rowIndex + 4] * velDiff.y +
				LayerValues0[rowIndex + 5] * velDiff.z;
		}

		layer1out[outputIndex] = max(0.0, val + LayerValues0[bias1offset + outputIndex]);
	}

	//Layer 2
	float[LAYER_SIZE] layer2out;
	//1x32 @ 32x32 = 1x32;		ACTUALLY 1x32 @ 32x32
	//output is array[32] where each array element is computed with 2 row vectors 
	for(int outputIndex = 0; outputIndex < LAYER_SIZE; outputIndex++)
	{
		float val = 0;
		uint rowIndex = layer2offset + LAYER_SIZE * outputIndex;

		#pragma unroll
		for(int i = 0; i < LAYER_SIZE; i++)
			val += layer1out[i] * LayerValues0[rowIndex + i];

		layer2out[outputIndex] = max(0.0, val + LayerValues0[bias2offset + outputIndex]);
	}

	//Layer 3
	//1x32 @ 32xOUT = 1xOUT;	ACTUALLY 1x32 @ OUTx32
	//output is array[MusclesCount] where each array element is 
	for(int outputIndex = 0; outputIndex < MusclesCount; outputIndex++)
	{
		float  val = 0;
		uint rowIndex = layer3offset + LAYER_SIZE * outputIndex;

		#pragma unroll
		for(int i = 0; i < LAYER_SIZE; i++)
			val += layer2out[i] * LayerValues0[rowIndex + i];

		float defaultLength = DefaultMuscleLengths[outputIndex / 4][outputIndex % 4];
		float maxOffset = defaultLength * MAX_MUSCLE_OFFSET;
		MuscleLengths0[musclesOffset + outputIndex] = clamp
		(
			MuscleLengths0[musclesOffset + outputIndex] + val + LayerValues0[bias3offset+outputIndex], 
			defaultLength - maxOffset, 
			defaultLength + maxOffset
		);
	}

};

void updateLengths1()
{
	//2 arrays of vec3 + 2 vec3
	const uint N = 2 * 3 * PointsCount;
	
	const uint layer1 = N * LAYER_SIZE;
	const uint layer2 = LAYER_SIZE * LAYER_SIZE;
	const uint layer3 = LAYER_SIZE * MusclesCount;
	const uint bias1 = LAYER_SIZE;
	const uint bias2 = LAYER_SIZE;
	const uint bias3 = MusclesCount;

	const uint instanceOffset = uint(gl_GlobalInvocationID.x) * (layer1+layer2+layer3+bias1+bias2+bias3);
	const uint pointsOffset = uint(gl_GlobalInvocationID.x) * PointsCount;
	const uint musclesOffset = uint(gl_GlobalInvocationID.x) * MusclesCount;

	const uint layer1offset = instanceOffset;
	const uint bias1offset = instanceOffset + layer1;

	const uint layer2offset = instanceOffset + layer1 + bias1;
	const uint bias2offset = instanceOffset + layer1 + bias1 + layer2;

	const uint layer3offset = instanceOffset + layer1 + bias1 + layer2 + bias2;
	const uint bias3offset = instanceOffset + layer1 + bias1 + layer2 + bias2 + layer3;

	//Layer 1
	float[LAYER_SIZE] layer1out;
	Point referencePoint = Points0[pointsOffset];
	//1xIN @ INx32 = 1x32;		ACTUALLY 1xIN @ 32xIN
	//output is array[32] where each array element is computed with 2 row vectors

	for(int outputIndex = 0; outputIndex < LAYER_SIZE; outputIndex++)
	{
		uint rowOffset = layer1offset + N * outputIndex;
		float val =
		LayerValues1[rowOffset + 0] * referencePoint.Position.x +
		LayerValues1[rowOffset + 1] * referencePoint.Position.y +
		LayerValues1[rowOffset + 2] * referencePoint.Position.z +
		LayerValues1[rowOffset + 3] * referencePoint.Velocity.x +
		LayerValues1[rowOffset + 4] * referencePoint.Velocity.y +
		LayerValues1[rowOffset + 5] * referencePoint.Velocity.z;

		//#pragma unroll
		for(uint i = 1; i < PointsCount; i ++)
		{
			vec3 posDiff = Points0[pointsOffset + i].Position - referencePoint.Position;
			vec3 velDiff = Points0[pointsOffset + i].Velocity - referencePoint.Velocity;

			uint rowIndex = rowOffset + 6 * i;
			val +=
				LayerValues1[rowIndex + 0] * posDiff.x +
				LayerValues1[rowIndex + 1] * posDiff.y +
				LayerValues1[rowIndex + 2] * posDiff.z +
				LayerValues1[rowIndex + 3] * velDiff.x +
				LayerValues1[rowIndex + 4] * velDiff.y +
				LayerValues1[rowIndex + 5] * velDiff.z;
		}

		layer1out[outputIndex] = max(0.0, val + LayerValues1[bias1offset + outputIndex]);
	}

	//Layer 2
	float[LAYER_SIZE] layer2out;
	//1x32 @ 32x32 = 1x32;		ACTUALLY 1x32 @ 32x32
	//output is array[32] where each array element is computed with 2 row vectors 
	for(int outputIndex = 0; outputIndex < LAYER_SIZE; outputIndex++)
	{
		float val = 0;
		uint rowIndex = layer2offset + LAYER_SIZE * outputIndex;

		#pragma unroll
		for(int i = 0; i < LAYER_SIZE; i++)
			val += layer1out[i] * LayerValues1[rowIndex + i];

		layer2out[outputIndex] = max(0.0, val + LayerValues1[bias2offset + outputIndex]);
	}

	//Layer 3
	//1x32 @ 32xOUT = 1xOUT;	ACTUALLY 1x32 @ OUTx32
	//output is array[MusclesCount] where each array element is 
	for(int outputIndex = 0; outputIndex < MusclesCount; outputIndex++)
	{
		float  val = 0;
		uint rowIndex = layer3offset + LAYER_SIZE * outputIndex;

		#pragma unroll
		for(int i = 0; i < LAYER_SIZE; i++)
			val += layer2out[i] * LayerValues1[rowIndex + i];

		float defaultLength = DefaultMuscleLengths[outputIndex / 4][outputIndex % 4];
		float maxOffset = defaultLength * MAX_MUSCLE_OFFSET;
		MuscleLengths0[musclesOffset + outputIndex] = clamp
		(
			MuscleLengths0[musclesOffset + outputIndex] + val + LayerValues1[bias3offset+outputIndex], 
			defaultLength - maxOffset, 
			defaultLength + maxOffset
		);
	}

};

void main()
{
	if(LayerReadBuffer == 0) updateLengths0();
	else updateLengths1();
};