#pragma once
#include <vector>
#include <glm/glm.hpp>

#define SPIDER 0

#if SPIDER
	#define MAX_POINTS 9
#else
	#define MAX_POINTS 8
#endif


#define MAX_MUSCLES 32
#define MAX_MUSCLES_PER_POINT 4

#define MAX_BONES 32
#define MAX_BONES_PER_POINT 4

#define LAYER_SIZE 32

#define EPISODE_LENGTH 350

#define RIGHT_STD_OFFSET 0.7f
#define LEFT_STD_OFFSET 0.0f

struct Shape
{
	std::vector<glm::vec3> Points{};

	std::vector<glm::ivec2> Muscles{};
	std::vector<glm::ivec2> Bones{};
	
	std::vector<float> MusclesLength;
	std::vector<float> BoneLengths;
};
