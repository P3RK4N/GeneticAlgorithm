#include "ShapeRenderer.h"

#include <glad/glad.h>
#include <Tigraf/Renderer/Renderer.h>
#include <Tigraf/Input/glfwInput.h>

#include <execution>
#include <iostream>

Ref<RWBuffer> ShapeRenderer::s_PosVelBuffer0 = nullptr;
Ref<RWBuffer> ShapeRenderer::s_LengthBuffer0 = nullptr;
Ref<RWBuffer> ShapeRenderer::s_LayerBuffer0 = nullptr;
Ref<RWBuffer> ShapeRenderer::s_LayerBuffer1 = nullptr;
Ref<RWBuffer> ShapeRenderer::s_StatsBuffer0 = nullptr;
Ref<RWBuffer> ShapeRenderer::s_FitnessIndicesBuffer0 = nullptr;

uint32_t ShapeRenderer::s_CurrentFrame = 0;
uint32_t ShapeRenderer::s_EpisodeFrameLength = EPISODE_LENGTH;

uint32_t ShapeRenderer::s_MaxAmountOfInstances = 0;

Ref<UniformBuffer> ShapeRenderer::s_ShapeData = nullptr;
Ref<VertexBuffer> ShapeRenderer::s_ShapeVertexBuffer = nullptr;

Ref<Shader> ShapeRenderer::s_ShapeShader = nullptr;
Ref<Shader> ShapeRenderer::s_PhysicsCompute = nullptr;
Ref<Shader> ShapeRenderer::s_GeneticCompute = nullptr;
Ref<Shader> ShapeRenderer::s_FitnessCompute = nullptr;
Ref<Shader> ShapeRenderer::s_EvolutionCompute = nullptr;

void ShapeRenderer::FillBuffers
(
	const Shape& shape, 
	uint32_t rwPosVelBufferIndex, 
	uint32_t rwLengthBufferIndex,
	uint32_t rwLayerBufferIndex0,
	uint32_t rwLayerBufferIndex1,
	uint32_t rwStatsBufferIndex,
	uint32_t rwFitnessIndicesBufferIndex,
	uint32_t uniformBufferIndex,
	uint32_t maxAmountOfInstances
)
{
	ShapeRenderer::s_MaxAmountOfInstances = maxAmountOfInstances;

	//Filling RWBuffers for Points and Muscle Lengths
	struct Point
	{
		alignas(16) glm::vec3 Position = { 0, 0, 0 };
		alignas(16) glm::vec3 Velocity = { 0, 0, 0 };

		Point() = default;
		Point(const glm::vec3& pos) : Position(pos) {}
	};

	std::vector<Point> posVelData;
	std::vector<float> muscleData;
	posVelData.reserve(shape.Points.size() * maxAmountOfInstances);
	muscleData.reserve(shape.MusclesLength.size() * maxAmountOfInstances);

	for(uint32_t i = 0; i < maxAmountOfInstances; i++)
	{
		for(auto& pos : shape.Points) posVelData.emplace_back(pos);
		for(auto& muscleLength : shape.MusclesLength) muscleData.emplace_back(muscleLength);
	}

	ShapeRenderer::s_PosVelBuffer0 = RWBuffer::create(posVelData.data(), shape.Points.size() * sizeof(Point) * maxAmountOfInstances, GL_DYNAMIC_STORAGE_BIT);
	ShapeRenderer::s_PosVelBuffer0->bind(rwPosVelBufferIndex);

	ShapeRenderer::s_LengthBuffer0 = RWBuffer::create(muscleData.data(), shape.MusclesLength.size() * sizeof(float) * maxAmountOfInstances, GL_DYNAMIC_STORAGE_BIT);
	ShapeRenderer::s_LengthBuffer0->bind(rwLengthBufferIndex);

	//Filling RWBuffers for Layer data--------------------------------
	uint32_t N = 2 * 3 * shape.Points.size();
	uint32_t layer1size = N * LAYER_SIZE;
	uint32_t layer2size = LAYER_SIZE * LAYER_SIZE;
	uint32_t layer3size = LAYER_SIZE * shape.Muscles.size();
	uint32_t bias1size = LAYER_SIZE;
	uint32_t bias2size = LAYER_SIZE;
	uint32_t bias3size = shape.Muscles.size();
	uint32_t totalSize = (layer1size + layer2size + layer3size + bias1size + bias2size + bias3size);
	std::vector<float> layerValues;
	layerValues.resize(totalSize * maxAmountOfInstances, 0);

	float fac = sqrt(6 * N);
	std::srand((unsigned)time(NULL));
	for(auto& val : layerValues) val = (1.0f * std::rand() / RAND_MAX * 2.0f - 1.0f) * fac;

	ShapeRenderer::s_LayerBuffer0 = RWBuffer::create(layerValues.data(), layerValues.size()*sizeof(float), GL_DYNAMIC_STORAGE_BIT);
	ShapeRenderer::s_LayerBuffer1 = RWBuffer::create(layerValues.data(), layerValues.size()*sizeof(float), GL_DYNAMIC_STORAGE_BIT);
	ShapeRenderer::s_LayerBuffer0->bind(rwLayerBufferIndex0);
	ShapeRenderer::s_LayerBuffer1->bind(rwLayerBufferIndex1);

	//Filling rwBuffer for Stats data
	//Stats structure:
	//StatValues0[0] =>	instances count;
	//StatValues0[1] => distances sum;
	//StatValues0[2] => squared distances sum;
	//StatValues0[3] => mean;
	//StatValues0[4] => variance;
	//StatValues0[5] => LowLimit;
	//StatValues0[6] => HighLimit;
	//StatValues0[7] => LayerReadBuffer;
	//StatValues0[8] => distances[0]...
	std::vector<uint32_t> stats;
	stats.resize(8 + maxAmountOfInstances * 2, 0);
	stats[0] = maxAmountOfInstances;
	stats[7] = 0;
	for(int i = 0; i < maxAmountOfInstances; i++) stats[8 + 1 + 2 * i] = i;
	ShapeRenderer::s_StatsBuffer0 = RWBuffer::create(stats.data(), stats.size() * sizeof(float), GL_DYNAMIC_STORAGE_BIT);
	ShapeRenderer::s_StatsBuffer0->bind(rwStatsBufferIndex);

	//Filling rwFitnessIndicesBuffer
	ShapeRenderer::s_FitnessIndicesBuffer0 = RWBuffer::create(nullptr, sizeof(uint32_t) * maxAmountOfInstances, GL_DYNAMIC_STORAGE_BIT);
	ShapeRenderer::s_FitnessIndicesBuffer0->bind(rwFitnessIndicesBufferIndex);

	//Filling Uniform Buffer with constant values
	struct ShapeData
	{
		uint32_t LastFrame;
		uint32_t PointsCount;
		uint32_t MusclesCount;
		uint32_t BonesCount;

		glm::ivec2 Connections[MAX_MUSCLES + MAX_BONES];
		float BoneLengths[MAX_BONES];
		float DefaultMuscleLengths[MAX_MUSCLES];
		glm::vec4 DefaultPointPositions[MAX_POINTS];
	} shapeData{};

	std::vector<glm::ivec2> connections;
	connections.reserve(shape.Muscles.size() + shape.Bones.size());
	for(auto& muscle : shape.Muscles) connections.emplace_back(muscle);
	for(auto& bone : shape.Bones) connections.emplace_back(bone);

	shapeData.LastFrame = ShapeRenderer::s_CurrentFrame == 500;
	shapeData.PointsCount = shape.Points.size();
	shapeData.MusclesCount = shape.Muscles.size();
	shapeData.BonesCount = shape.Bones.size();
	memcpy(shapeData.Connections, connections.data(), connections.size() * sizeof(glm::ivec2));
	memcpy(shapeData.BoneLengths, shape.BoneLengths.data(), shape.BoneLengths.size() * sizeof(float));
	memcpy(shapeData.DefaultMuscleLengths, shape.MusclesLength.data(), shape.MusclesLength.size() * sizeof(float));
	
	for(int i = 0; i < shape.Points.size(); i++)
		shapeData.DefaultPointPositions[i] = { shape.Points[i], 1.0 };

	/*
	UNIFORM BUFFER LAYOUT:
	
	currentFrame;
	pointsCount;
	totalMusclesCount;
	totalBonesCount;

	boneConnections[MAX_MUSCLES + MAX_BONES];
	boneLengths[MAX_BONES];
	defaultMuscleLengths[MAX_MUSCLES];
	defaultPointPositions[MAX_POINTS];
	*/

	ShapeRenderer::s_ShapeData = UniformBuffer::create(&shapeData, sizeof(shapeData), GL_DYNAMIC_STORAGE_BIT);
	ShapeRenderer::s_ShapeData->bind(uniformBufferIndex);



	//Filling vertex buffer with pairs of indices for drawing
	struct ShapeVertex
	{
		glm::ivec2 connectionIndices;
		int isMuscle;

		ShapeVertex(const glm::ivec2& connection, int muscle) : connectionIndices(connection), isMuscle(muscle) {}
	};

	std::vector<ShapeVertex> vertexData;
	vertexData.reserve(shape.Muscles.size() + shape.Bones.size());

	for(auto& muscle : shape.Muscles) vertexData.emplace_back(muscle, 1);
	for(auto& bone : shape.Bones) vertexData.emplace_back(bone, 0);

	ShapeRenderer::s_ShapeVertexBuffer = VertexBuffer::create(vertexData.size(), sizeof(ShapeVertex), vertexData.data(), 0);
	ShapeRenderer::s_ShapeVertexBuffer->pushVertexAttribute(VertexAttributeType::INT2);
	ShapeRenderer::s_ShapeVertexBuffer->pushVertexAttribute(VertexAttributeType::INT);
}

void ShapeRenderer::InitShaders()
{
	ShapeRenderer::s_ShapeShader = Shader::create("resources\\custom_shaders\\ShapeShader.glsl");
	ShapeRenderer::s_PhysicsCompute = Shader::create("resources\\custom_shaders\\PhysicsCompute.glsl");
	ShapeRenderer::s_GeneticCompute = Shader::create("resources\\custom_shaders\\GeneticCompute.glsl");
	ShapeRenderer::s_FitnessCompute = Shader::create("resources\\custom_shaders\\FitnessCompute.glsl");
	ShapeRenderer::s_EvolutionCompute = Shader::create("resources\\custom_shaders\\EvolutionCompute.glsl");
}

void ShapeRenderer::DrawInstances(uint32_t numInstances)
{
	TIGRAF_ASSERT(numInstances <= ShapeRenderer::s_MaxAmountOfInstances, "Invalid instance amount");

	glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
	ShapeRenderer::s_ShapeShader->use();
	Renderer::s_RendererAPI->drawPointsInstanced(ShapeRenderer::s_ShapeVertexBuffer, numInstances);
}

void ShapeRenderer::ComputePhysics(int x, int y, int z)
{
	TIGRAF_ASSERT(ShapeRenderer::s_PhysicsCompute, "Physics shader not initialized!");

	ShapeRenderer::s_CurrentFrame = (ShapeRenderer::s_CurrentFrame + 1) % ShapeRenderer::s_EpisodeFrameLength;
	ShapeRenderer::s_ShapeData->updateBuffer(&ShapeRenderer::s_CurrentFrame, 4, 0);

	glMemoryBarrier(GL_UNIFORM_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
	ShapeRenderer::s_PhysicsCompute->dispatch(x, y, z);

	if(!ShapeRenderer::s_CurrentFrame) ShapeRenderer::SetStats(x, y, z);
}

void ShapeRenderer::ComputeMovement(int x, int y, int z)
{
	TIGRAF_ASSERT(ShapeRenderer::s_GeneticCompute, "Genetic shader not initialized!");

	
	glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
	ShapeRenderer::s_GeneticCompute->dispatch(x, y, z);
}

void ShapeRenderer::ComputeFitness(int x, int y, int z)
{
	TIGRAF_ASSERT(ShapeRenderer::s_FitnessCompute, "Fitness shader not initialized!");

	
	glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
	ShapeRenderer::s_FitnessCompute->dispatch(x, y, z);
}

void ShapeRenderer::ComputeFitnessCPU()
{
	std::vector<std::pair<float,uint32_t>> distances_indices;
	distances_indices.resize(ShapeRenderer::s_MaxAmountOfInstances, {-1,-1});

	ShapeRenderer::s_StatsBuffer0->get(distances_indices.data(), sizeof(float)*8, sizeof(std::pair<float,uint32_t>)*ShapeRenderer::s_MaxAmountOfInstances);


	std::sort(std::execution::par, distances_indices.begin(), distances_indices.end(), [&](const std::pair<float, uint32_t>& first, const std::pair<float, uint32_t>& second)
	{
		return first.first < second.first;
	});

	std::vector<uint32_t> sortedIndices = {};
	sortedIndices.reserve(ShapeRenderer::s_MaxAmountOfInstances);

	for(auto& [dist, i] : distances_indices)
	{
		sortedIndices.push_back(i);
	};

	INFO(std::format("Best distance: {0}", distances_indices[ShapeRenderer::s_MaxAmountOfInstances-1].first));

	ShapeRenderer::s_FitnessIndicesBuffer0->updateBuffer(sortedIndices.data(), sortedIndices.size() * sizeof(uint32_t), 0);
}

void ShapeRenderer::ComputeEvolution(int x, int y, int z)
{
	TIGRAF_ASSERT(ShapeRenderer::s_EvolutionCompute, "Evolution shader not initialized!");

	
	glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
	ShapeRenderer::s_EvolutionCompute->dispatch(x, y, z);
}

void ShapeRenderer::SetStats(int x, int y, int z)
{
	glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
	struct SubStats
	{
		uint32_t InstancesCount;
		float DistancesSum;
		float SquaredDistancesSum;
		float DistancesMean;
		float DistancesVariance;
		float LowLimit;
		float HighLimit;
		uint32_t LayerReadBuffer;
	} subStats;

	ShapeRenderer::s_StatsBuffer0->get(&subStats, 0, sizeof(subStats));

	subStats.DistancesMean = subStats.DistancesSum / subStats.InstancesCount;
	subStats.DistancesVariance = subStats.SquaredDistancesSum / subStats.InstancesCount - subStats.DistancesMean * subStats.DistancesMean;
	subStats.LowLimit = subStats.DistancesMean - sqrt(subStats.DistancesVariance) * LEFT_STD_OFFSET;
	subStats.HighLimit = subStats.DistancesMean + sqrt(subStats.DistancesVariance) * RIGHT_STD_OFFSET;

	INFO(std::format("Instances Count: {0}", subStats.InstancesCount));

	subStats.DistancesSum = 0;
	subStats.SquaredDistancesSum = 0;
	subStats.LayerReadBuffer ^= 1;

	ShapeRenderer::s_StatsBuffer0->updateBuffer(&subStats, sizeof(subStats), 0);

	ShapeRenderer::ComputeFitnessCPU();

	ShapeRenderer::ComputeEvolution(x, y, z);

	static uint32_t iter = 0;
	INFO(std::format("Iteration {0} ^^^^^^^^^^^^^^^^^^^^", ++iter));
}
