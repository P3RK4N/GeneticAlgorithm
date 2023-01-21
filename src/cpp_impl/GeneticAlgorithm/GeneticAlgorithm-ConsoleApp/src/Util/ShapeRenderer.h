#pragma once
#include "Shape.h"

#include <Tigraf/Renderer/Buffers/Buffer.h>
#include <Tigraf/Renderer/Shaders/Shader.h>
#include <Tigraf/Core/Core.h>

#include <glad/glad.h>

using namespace Tigraf;

class ShapeRenderer
{
public:
	/*
	* Fills RWBuffers at specified indices with data of 'amountOfInstances' shapes.
	* 
	* @param shape The shape we want to draw
	* @param rwPosVelBufferIndex Index of RWBuffer. Use define 'RW_BUFFER_x' for index 'x'
	* @param rwLengthBufferIndex Buffer for storing current muscle lengths
	* @param rwLayerBufferIndex Buffer for storing neural layers of instances
	* @param rwStatsBufferIndex Buffer for statistics
	* @param uniformBufferIndex is used for storinf constant data. Use define 'UNIFORM_BUFFER_x' for index 'x'
	* @param maxAmountOfInstances number of shape instances
	*/
	static void FillBuffers
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
	);

	static void InitShaders();

	static void DrawInstances(uint32_t instances);
	static void ComputePhysics(int x, int y, int z);
	static void ComputeMovement(int x, int y, int z);

protected:
	static void SetStats(int x, int y, int z);
	static void ComputeFitness		(int x, int y, int z);
	static void ComputeFitnessCPU	();
	static void ComputeEvolution(int x, int y, int z);

	static uint32_t s_CurrentFrame;
	static uint32_t s_EpisodeFrameLength;

	//Buffers which store position and velocity data. 
	//They are used interchargeably so that we dont need to send data to CPU between frames

	static Ref<RWBuffer> s_PosVelBuffer0;

	static Ref<RWBuffer> s_LengthBuffer0;

	//Buffer for storing layer data
	static Ref<RWBuffer> s_LayerBuffer0;
	static Ref<RWBuffer> s_LayerBuffer1;

	//Buffer for statistics of the episode
	static Ref<RWBuffer> s_StatsBuffer0;

	static Ref<RWBuffer> s_FitnessIndicesBuffer0;

	//Buffer for storing constant data
	static Ref<UniformBuffer> s_ShapeData;

	//Buffer which stores instance index and index of connection
	static Ref<VertexBuffer> s_ShapeVertexBuffer;

	//Used for drawing shapes on screen
	static Ref<Shader> s_ShapeShader;
	static uint32_t s_MaxAmountOfInstances;

	//Used for physics
	static Ref<Shader> s_PhysicsCompute;

	//Used for genetic algorithm
	static Ref<Shader> s_GeneticCompute;

	//Used for finding fitness indexes of good instances
	static Ref<Shader> s_FitnessCompute;

	//Used for evolving generations
	static Ref<Shader> s_EvolutionCompute;
};