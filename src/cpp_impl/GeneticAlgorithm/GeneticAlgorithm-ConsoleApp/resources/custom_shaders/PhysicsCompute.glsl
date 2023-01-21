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

layout(std430, binding = 0) buffer PointBuffer0
{
	Point[] Points0;
};

layout(std430, binding = 2) buffer LengthBuffer0
{
	float[] MuscleLengths0;
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


#define GROUND_FORCE 0.002
#define DAMPING_COEFF_GROUND 2

#define GRAVITY_STRENGTH 0.002
#define COEFF_FRICTION 1

#define MUSCLE_DAMPNESS 0.1
#define MUSCLE_SPRINGINESS 0.05

#define BONE_DAMPNESS 0.1
#define BONE_SPRINGINESS 0.1

void main()
{
	uint invocationOffset = uint(gl_GlobalInvocationID.x);
	uint pointOffset = invocationOffset * PointsCount;
	uint muscleOffset = invocationOffset * MusclesCount;

	if(CurrentFrame == 0)
	{
		vec2 instanceOffset = vec2(0, 0);
		Point emptyPoint;
		emptyPoint.Velocity = vec3(0.0, 0.0, 0.0);

		#pragma unroll
		for(uint i = 0; i < MusclesCount; i++)
		{
			MuscleLengths0[muscleOffset + i] = DefaultMuscleLengths[i / 4][i % 4];
		}


		#pragma unroll
		for(int i = 0; i < PointsCount; i++)
		{
			instanceOffset += Points0[pointOffset + i].Position.xz;

			emptyPoint.Position = DefaultPointPositions[i].xyz;
			Points0[pointOffset + i] = emptyPoint;
		}

		instanceOffset /= PointsCount;
		float len = length(instanceOffset);

		Distance_Index di;
		di.dist = len;
		di.ind = invocationOffset;
		Distances[invocationOffset] = di;

		return;
	}
	
	vec3[MAX_POINTS] force;
	vec3[MAX_POINTS] groundForce;
	Point[MAX_POINTS] points;

	#pragma unroll
	for(int i = 0; i < PointsCount; i++)
	{
		force[i] = vec3(0.0, -GRAVITY_STRENGTH, 0.0);

		Point p = Points0[pointOffset + i];
		points[i] = p;

		groundForce[i].y = max(0.0, -p.Position.y);
		groundForce[i].y += smoothstep(-0.3,0.3, -p.Position.y) * -p.Velocity.y * DAMPING_COEFF_GROUND;

		force[i].y += groundForce[i].y;
	}

	for(int connectionIndex = 0; connectionIndex < MusclesCount + BonesCount; connectionIndex++)
	{
		uint indexA = Connections[connectionIndex / 2][(connectionIndex % 2) * 2 + 0];
		uint indexB = Connections[connectionIndex / 2][(connectionIndex % 2) * 2 + 1];

		Point pointA = points[indexA];
		Point pointB = points[indexB];

		vec3 AB = pointA.Position - pointB.Position;

		float len_AB = length(AB);
		vec3 dir_AB = normalize(AB);

		vec3 dampForce = - (pointA.Velocity - pointB.Velocity);
		vec3 springForce = dir_AB;
		
		if(connectionIndex < MusclesCount)
		{
			dampForce *= MUSCLE_DAMPNESS;
			springForce *= MUSCLE_SPRINGINESS * (len_AB - MuscleLengths0[muscleOffset + connectionIndex]);
		}
		else 
		{
			dampForce *= BONE_DAMPNESS;
			springForce *= BONE_SPRINGINESS * (len_AB - BoneLengths[(connectionIndex - MusclesCount) / 4][(connectionIndex - MusclesCount) % 4]);
		}

		force[indexA] += dampForce - springForce;
		force[indexB] -= dampForce - springForce;
	}

	#pragma unroll
	for(int i = 0; i < PointsCount; i++)
	{
		vec3 ev = points[i].Velocity + force[i];
		vec3 frictionForce = normalize(ev) * min(groundForce[i].y * COEFF_FRICTION, length(ev));
		force[i] -= frictionForce;
	
		points[i].Velocity += force[i];
		points[i].Position += points[i].Velocity;
		Points0[pointOffset + i] = points[i];
	}
};