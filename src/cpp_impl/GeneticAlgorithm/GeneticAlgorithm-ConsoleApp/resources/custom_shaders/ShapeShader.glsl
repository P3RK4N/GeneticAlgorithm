$VertexShader
#version 460 core

layout(location = 0) in ivec2 ConnectionIndicesVS;
layout(location = 1) in int isMuscleVS;

out flat ivec2 ConnectionIndicesGS;
out flat int isMuscleGS;
out flat uint instanceID;

void main()
{
	ConnectionIndicesGS = ConnectionIndicesVS;
	isMuscleGS = isMuscleVS;
	instanceID = gl_InstanceID;
}



$GeometryShader
#version 460 core

layout(points) in;
layout(triangle_strip, max_vertices = 16) out;

struct Point
{
	vec3 Position;
	vec3 Velocity;
};

#define DRAW_MOD 128

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
	Point[] points0;
};

layout(std430, binding = 2) buffer LengthBuffer0
{
	float[] MuscleLengths0;
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

layout(std140, binding = 1) uniform PerFrameBuffer
{
	mat4 CameraViewProjection;
	vec3 CameraWorldPosition;
	float TotalTime;
	float FrameTime;
};

in flat ivec2 ConnectionIndicesGS[];
in flat int isMuscleGS[];
in flat uint instanceID[];

out flat ivec2 ConnectionIndicesPS;
out flat int isMusclePS;
out vec3 NormalPS;
out flat uint instanceIDPS;

void main()
{
	isMusclePS = isMuscleGS[0];
	ConnectionIndicesPS = ConnectionIndicesGS[0];
	instanceIDPS = instanceID[0];

	uint offset = instanceID[0] * PointsCount;

	Point firstPoint, secondPoint;

	firstPoint = points0[offset + ConnectionIndicesGS[0].x];
	secondPoint = points0[offset + ConnectionIndicesGS[0].y];

	vec3 instanceOffset = vec3(float(instanceID[0] / DRAW_MOD) * 3.0, 0, float(instanceID[0] % DRAW_MOD) * 3.0);
	vec3 pos0 = firstPoint.Position + instanceOffset;
	vec3 pos1 = secondPoint.Position + instanceOffset;

	float cameraDistance = distance(pos0, CameraWorldPosition);

	if(cameraDistance > 200) return;

	vec3 dir = normalize(pos0 - pos1);
	float diff = float(isMuscleGS[0]) * 0.01;

	if(cameraDistance > 10)
	{
		vec3 cameraVector = normalize(- pos0 + CameraWorldPosition);
		vec3 normal = normalize(cross(dir, cameraVector));
		vec3 smallNormal = normal * (0.03 - diff);

		vec4[4] vertices =
		{
			CameraViewProjection * vec4(pos0 - smallNormal, 1.0),
			CameraViewProjection * vec4(pos0 + smallNormal, 1.0),
			CameraViewProjection * vec4(pos1 - smallNormal, 1.0),
			CameraViewProjection * vec4(pos1 + smallNormal, 1.0)
		};

		NormalPS = cross(normal, dir);

		gl_Position = vertices[0];
		EmitVertex();

		gl_Position = vertices[1];
		EmitVertex();

		gl_Position = vertices[2];
		EmitVertex();

		gl_Position = vertices[3];
		EmitVertex();
	}
	else
	{
		vec3 normal = normalize(cross(dir, vec3(0.0, 1.0, 0.0)));
		vec3 tangent = normalize(cross(dir, normal));
	
		vec3 smallNormal = normal * (0.03 - diff);
		vec3 smallTangent = tangent * (0.03 - diff);

		vec4[8] vertices =
		{
			CameraViewProjection * vec4(pos1 - smallTangent - smallNormal, 1.0),
			CameraViewProjection * vec4(pos1 + smallTangent - smallNormal, 1.0),
			CameraViewProjection * vec4(pos0 - smallTangent - smallNormal, 1.0),
			CameraViewProjection * vec4(pos0 + smallTangent - smallNormal, 1.0),
			CameraViewProjection * vec4(pos0 - smallTangent + smallNormal, 1.0),
			CameraViewProjection * vec4(pos0 + smallTangent + smallNormal, 1.0),
			CameraViewProjection * vec4(pos1 - smallTangent + smallNormal, 1.0),
			CameraViewProjection * vec4(pos1 + smallTangent + smallNormal, 1.0)
		};

		NormalPS = -normal;

		gl_Position = vertices[0];
		EmitVertex();
	
		gl_Position = vertices[1];
		EmitVertex();
	
		gl_Position = vertices[2];
		EmitVertex();
	
		gl_Position = vertices[3];
		EmitVertex();

		NormalPS = -dir;
	
		gl_Position = vertices[4];
		EmitVertex();

		gl_Position = vertices[5];
		EmitVertex();

		NormalPS = normal;

		gl_Position = vertices[6];
		EmitVertex();

		gl_Position = vertices[7];
		EmitVertex();

		NormalPS = dir;

		gl_Position = vertices[1];
		EmitVertex();

		NormalPS = tangent;

		gl_Position = vertices[5];
		EmitVertex();

		gl_Position = vertices[3];
		EmitVertex();

		EndPrimitive();

		NormalPS = dir;

		gl_Position = vertices[1];
		EmitVertex();

		gl_Position = vertices[6];
		EmitVertex();

		gl_Position = vertices[0];
		EmitVertex();

		NormalPS = -tangent;

		gl_Position = vertices[4];
		EmitVertex();

		gl_Position = vertices[2];
		EmitVertex();
	}
}

$PixelShader
#version 460 core

#define SPIDER 0

#if SPIDER
	#define MAX_POINTS 9
#else
	#define MAX_POINTS 8
#endif

#define MAX_BONES 32
#define MAX_MUSCLES 32


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

in flat ivec2 ConnectionIndicesPS;
in flat int isMusclePS;
in flat vec3 NormalPS;
in flat uint instanceIDPS;

out vec4 FragColor;

void main()
{
	vec4 col = 
		isMusclePS		* vec4(0.8, 0.3, 0.1, 0.3) + 
		(isMusclePS^1)	* vec4(0.72, 0.76, 0.8, 1.0);

	float light = max(0.2, dot(NormalPS, vec3(0.0, 1.0, 0.0)) / 2.0 + 0.5);


	FragColor = vec4(col.rgb * light, col.a);
}